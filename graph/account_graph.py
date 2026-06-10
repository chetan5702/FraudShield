"""
account_graph.py
FraudShield — Account Similarity Graph (Real Dataset)

Since the hackathon dataset is account-level (not transaction-level),
we build a similarity graph instead of a transaction graph:

  Nodes = accounts (9,082)
  Edges = cosine similarity between account feature vectors above a threshold

Accounts with unusually similar behavioral profiles are connected —
this surfaces coordinated fraud rings, mule networks, and synthetic
identity clusters that transaction-level graphs would miss entirely.

Mule/fraud signals derived from graph:
  1. High degree centrality     — connected to many similar accounts
  2. Clique membership          — part of a tight cluster of similar accounts
  3. Community anomaly score    — account's community has high avg fraud proxy rate
  4. Isolation score            — genuinely isolated accounts (no similar peers)

Run standalone:
    python graph/account_graph.py
Or import:
    from graph.account_graph import run_account_graph
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")

# Similarity threshold — accounts more similar than this get an edge
# Higher = fewer edges (sparser, faster); Lower = more edges (denser, slower)
SIMILARITY_THRESHOLD = 0.92
PCA_COMPONENTS       = 50    # reduce 3924 → 50 dims before similarity (speed + noise reduction)
MAX_ACCOUNTS         = 9082  # process all accounts


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_real_data() -> tuple:
    """Load processed real dataset and full feature matrix."""
    meta_path = os.path.join(PROCESSED_DIR, "real_dataset_processed.csv")
    feat_path = os.path.join(PROCESSED_DIR, "real_dataset_features.csv")

    if not os.path.exists(meta_path) or not os.path.exists(feat_path):
        raise FileNotFoundError(
            "Real dataset not found. Run first:\n"
            "  py ingestion/load_real_dataset.py --input path/to/DataSet.csv"
        )

    meta_df = pd.read_csv(meta_path)
    feat_df = pd.read_csv(feat_path)

    print(f"[Load]  Accounts: {len(meta_df):,} | Features: {feat_df.shape[1]:,}")
    return meta_df, feat_df


# ── Feature Preparation ───────────────────────────────────────────────────────

def prepare_features(feat_df: pd.DataFrame, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    """
    Standardize + PCA-reduce feature matrix.
    PCA serves two purposes:
      1. Speed — cosine similarity on 50 dims is 78x faster than 3924
      2. Denoising — removes low-variance noise dimensions
    """
    print(f"[PCA]   Reducing {feat_df.shape[1]} → {n_components} dimensions...")

    X = feat_df.fillna(feat_df.median()).values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[PCA]   Explained variance: {explained:.1f}%")
    return X_reduced


# ── Graph Construction ────────────────────────────────────────────────────────

def build_similarity_graph(
    X: np.ndarray,
    account_ids: list,
    threshold: float = SIMILARITY_THRESHOLD,
    batch_size: int = 500,
) -> nx.Graph:
    """
    Build undirected similarity graph in batches.
    Edges connect accounts with cosine similarity >= threshold.
    Batched to avoid OOM on large datasets.
    """
    n = len(account_ids)
    G = nx.Graph()
    G.add_nodes_from(account_ids)

    print(f"[Graph] Building similarity graph (threshold={threshold})...")
    edge_count = 0

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = X[start:end]

        # Similarity between this batch and all accounts
        sim_matrix = cosine_similarity(batch, X)

        for i, row_sim in enumerate(sim_matrix):
            global_i = start + i
            # Only look at accounts after global_i to avoid duplicate edges
            for j in range(global_i + 1, n):
                if row_sim[j] >= threshold:
                    G.add_edge(
                        account_ids[global_i],
                        account_ids[j],
                        weight=float(row_sim[j])
                    )
                    edge_count += 1

        if (start // batch_size) % 5 == 0:
            print(f"[Graph] Processed {end:,}/{n:,} accounts | Edges so far: {edge_count:,}")

    print(f"[Graph] Done — {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ── Graph Feature Extraction ──────────────────────────────────────────────────

def compute_graph_features(
    G: nx.Graph,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract node-level graph features for each account:
      - degree              : number of similar accounts connected
      - degree_centrality   : normalized degree
      - clustering_coeff    : how tightly connected neighbors are (clique-ness)
      - component_size      : size of connected component account belongs to
      - component_fraud_rate: avg fraud proxy rate in account's component
      - is_isolated         : account has no similar peers
    """
    print("[Features] Computing graph features...")

    fraud_map = dict(zip(meta_df["account_id"], meta_df["is_fraud"]))

    # Degree
    degrees    = dict(G.degree())
    deg_cent   = nx.degree_centrality(G)
    clustering = nx.clustering(G)

    # Connected components
    components = list(nx.connected_components(G))
    comp_map   = {}  # account → component index
    for idx, comp in enumerate(components):
        for acc in comp:
            comp_map[acc] = idx

    comp_sizes      = {idx: len(comp) for idx, comp in enumerate(components)}
    comp_fraud_rate = {}
    for idx, comp in enumerate(components):
        fraud_vals = [fraud_map.get(acc, 0) for acc in comp]
        comp_fraud_rate[idx] = np.mean(fraud_vals) if fraud_vals else 0.0

    rows = []
    for acc in meta_df["account_id"]:
        comp_idx = comp_map.get(acc, -1)
        rows.append({
            "account_id":          acc,
            "degree":              degrees.get(acc, 0),
            "degree_centrality":   deg_cent.get(acc, 0.0),
            "clustering_coeff":    clustering.get(acc, 0.0),
            "component_size":      comp_sizes.get(comp_idx, 1),
            "component_fraud_rate":comp_fraud_rate.get(comp_idx, 0.0),
            "is_isolated":         1 if degrees.get(acc, 0) == 0 else 0,
        })

    return pd.DataFrame(rows)


# ── Composite Graph Risk Score ────────────────────────────────────────────────

def compute_graph_risk(graph_feat_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine graph features into a graph_risk_score [0-1] per account.

    Weighting rationale:
      30% degree centrality     — many similar peers → potential coordinated ring
      25% clustering coeff      — tight clique → organized fraud network
      25% component fraud rate  — guilt by association (community-level signal)
      20% component size        — large clusters of similar accounts are suspicious
    """
    df = graph_feat_df.copy()

    # Normalize component_size to [0,1]
    max_size = df["component_size"].max()
    df["comp_size_norm"] = df["component_size"] / max(max_size, 1)

    graph_risk = (
        0.30 * df["degree_centrality"] +
        0.25 * df["clustering_coeff"] +
        0.25 * df["component_fraud_rate"] +
        0.20 * df["comp_size_norm"]
    ).clip(0, 1)

    df["graph_risk_score"] = graph_risk.round(4)
    df["graph_label"] = df["graph_risk_score"].apply(
        lambda s: "HIGH_RISK" if s >= 0.65 else "SUSPECT" if s >= 0.35 else "CLEAN"
    )

    # Merge with meta
    base_cols = ["account_id", "account_type", "risk_rating", "occupation", "is_fraud"]
    optional  = [c for c in ["risk_score", "na_fraction", "row_std"] if c in meta_df.columns]
    result = meta_df[base_cols + optional].merge(df, on="account_id", how="left")

    result["graph_risk_score"] = result["graph_risk_score"].fillna(0)
    result["graph_label"]      = result["graph_label"].fillna("CLEAN")

    return result.sort_values("graph_risk_score", ascending=False).reset_index(drop=True)


# ── Cluster Summary ───────────────────────────────────────────────────────────

def summarize_clusters(G: nx.Graph, result_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return a summary of the top-N largest suspicious clusters."""
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    risk_col = "graph_risk_score" if "graph_risk_score" in result_df.columns else "is_fraud"
    risk_map   = dict(zip(result_df["account_id"], result_df[risk_col]))
    fraud_map  = dict(zip(result_df["account_id"], result_df["is_fraud"]))
    rating_map = dict(zip(result_df["account_id"], result_df["risk_rating"]))

    rows = []
    for i, comp in enumerate(components[:top_n]):
        comp = list(comp)
        rows.append({
            "cluster_id":       i + 1,
            "size":             len(comp),
            "avg_graph_risk":   round(np.mean([risk_map.get(a, 0) for a in comp]), 4),
            "fraud_rate":       round(np.mean([fraud_map.get(a, 0) for a in comp]), 3),
            "su_rating_count":  sum(1 for a in comp if rating_map.get(a) == "SU"),
            "sample_accounts":  ", ".join(comp[:3]),
        })
    return pd.DataFrame(rows)


def get_account_neighborhood(G: nx.Graph, account_id: str, depth: int = 1) -> nx.Graph:
    """Extract subgraph around an account for case investigation view."""
    nodes = {account_id}
    for _ in range(depth):
        nbrs = set()
        for n in nodes:
            nbrs.update(G.neighbors(n))
        nodes.update(nbrs)
    return G.subgraph(nodes).copy()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_account_graph(save: bool = True) -> tuple:
    print("=" * 55)
    print("  FraudShield — Account Similarity Graph")
    print("=" * 55)

    meta_df, feat_df = load_real_data()

    # Limit for speed if needed
    meta_df = meta_df.head(MAX_ACCOUNTS).reset_index(drop=True)
    feat_df = feat_df.head(MAX_ACCOUNTS).reset_index(drop=True)

    # Prepare features
    X = prepare_features(feat_df, n_components=PCA_COMPONENTS)

    # Build graph
    account_ids = meta_df["account_id"].tolist()
    G = build_similarity_graph(X, account_ids, threshold=SIMILARITY_THRESHOLD)

    # Extract features
    graph_feat_df = compute_graph_features(G, meta_df)
    result_df     = compute_graph_risk(graph_feat_df, meta_df)

    # Cluster summary
    cluster_summary = summarize_clusters(G, result_df)

    # Stats
    n_high    = (result_df["graph_label"] == "HIGH_RISK").sum()
    n_suspect = (result_df["graph_label"] == "SUSPECT").sum()
    n_comps   = nx.number_connected_components(G)
    largest   = max((len(c) for c in nx.connected_components(G)), default=0)

    print(f"\n[Results]")
    print(f"  Accounts analyzed    : {len(result_df):,}")
    print(f"  Graph edges          : {G.number_of_edges():,}")
    print(f"  Connected components : {n_comps:,}")
    print(f"  Largest cluster      : {largest:,} accounts")
    print(f"  HIGH_RISK            : {n_high:,}")
    print(f"  SUSPECT              : {n_suspect:,}")

    print(f"\n[Top Clusters]")
    print(cluster_summary.to_string(index=False))

    if save:
        out = os.path.join(PROCESSED_DIR, "account_graph_scores.csv")
        result_df.to_csv(out, index=False)
        print(f"\n[Saved]  → {out}")

    print("=" * 55)
    return result_df, G


if __name__ == "__main__":
    result_df, G = run_account_graph()

    print("\nTop 10 highest graph-risk accounts:")
    print(result_df[[
        "account_id", "account_type", "risk_rating",
        "graph_risk_score", "graph_label", "degree",
        "clustering_coeff", "component_fraud_rate"
    ]].head(10).to_string(index=False))
