"""
graph_detector.py
FraudShield — Graph-Based Mule Account Detection

Builds an incremental directed transaction graph where:
  nodes = accounts
  edges = money transfers (weighted by amount)

Mule accounts are identified by four signals:
  1. High betweenness centrality  — bridges between many accounts
  2. Rapid pass-through ratio     — receives and forwards money quickly
  3. Fan-in / fan-out pattern     — many senders → few receivers (or vice versa)
  4. Govt alert cross-reference   — account appears in regulatory alert data

Each account gets a mule_score [0–1] and a mule_label (CLEAN / SUSPECT / HIGH_RISK).
Graph is updated incrementally — only affected nodes/edges recomputed on new transactions.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")

PASS_THROUGH_WINDOW_HOURS = 2     # money in → out within this window = pass-through
MULE_THRESHOLD_SUSPECT    = 0.40
MULE_THRESHOLD_HIGH_RISK  = 0.65


# ── Incremental Graph Builder ─────────────────────────────────────────────────

class TransactionGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self._edge_buffer = []          # buffer for incremental updates

    def add_transactions(self, df: pd.DataFrame):
        """Add a batch of transactions. Only new nodes/edges are processed."""
        # Aggregate by (sender, receiver) pair before touching the graph —
        # avoids Python-level row iteration which is the main bottleneck.
        agg = (
            df.groupby(["sender_id", "receiver_id"])
            .agg(total_amount=("amount", "sum"), tx_count=("amount", "count"),
                 timestamps=("timestamp", list))
            .reset_index()
        )
        new_edges = 0
        for row in agg.itertuples(index=False):
            s, r = row.sender_id, row.receiver_id
            if not self.G.has_edge(s, r):
                self.G.add_edge(s, r, total_amount=0.0, tx_count=0, timestamps=[])
                new_edges += 1
            self.G[s][r]["total_amount"] += row.total_amount
            self.G[s][r]["tx_count"]     += row.tx_count
            self.G[s][r]["timestamps"].extend(row.timestamps)

        print(f"[Graph]  Nodes: {self.G.number_of_nodes():,}  |  "
              f"Edges: {self.G.number_of_edges():,}  |  New edges added: {new_edges:,}")

    def incremental_update(self, new_df: pd.DataFrame):
        """Add only new transactions without rebuilding the full graph."""
        self.add_transactions(new_df)


# ── Feature Extraction per Account ───────────────────────────────────────────

def compute_betweenness(G: nx.DiGraph, sample_k: int = 500) -> dict:
    """Approximate betweenness centrality (fast for large graphs)."""
    print("[Graph]  Computing betweenness centrality (approx)...")
    bc = nx.betweenness_centrality(G, k=min(sample_k, len(G)), normalized=True)
    return bc


def compute_pass_through_ratio(df: pd.DataFrame, window_hours: int = PASS_THROUGH_WINDOW_HOURS) -> dict:
    """
    For each account: fraction of received funds forwarded out within the time window.
    High ratio = mule-like behaviour.

    Vectorised with merge_asof — O(n log n) instead of O(n²).
    For each inflow event, we find the nearest outflow from the same account
    that occurs within [0, window_hours] after it, then sum the matched amounts.
    """
    df = df.sort_values("timestamp").copy()
    window = pd.Timedelta(hours=window_hours)

    # Build tidy inflow / outflow tables
    inflows = (
        df[["receiver_id", "timestamp", "amount"]]
        .rename(columns={"receiver_id": "account", "amount": "in_amt"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    outflows = (
        df[["sender_id", "timestamp", "amount"]]
        .rename(columns={"sender_id": "account", "amount": "out_amt"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # For each inflow, find the first outflow by the same account within the window.
    # merge_asof matches on timestamp (nearest-forward) per account group.
    matched = pd.merge_asof(
        inflows,
        outflows.rename(columns={"timestamp": "out_ts"}),
        left_on="timestamp",
        right_on="out_ts",
        by="account",
        direction="forward",
        tolerance=window,
    )

    # Amount actually passed through = min(in, out) for matched pairs
    matched["passed"] = np.where(
        matched["out_amt"].notna(),
        np.minimum(matched["in_amt"], matched["out_amt"]),
        0.0,
    )

    summary = matched.groupby("account").agg(
        total_in=("in_amt", "sum"),
        total_passed=("passed", "sum"),
    )
    summary["ratio"] = (summary["total_passed"] / summary["total_in"].clip(lower=1)).clip(upper=1.0)

    # Accounts that only appear as senders (no inflows) get ratio 0
    all_accounts = set(df["sender_id"]) | set(df["receiver_id"])
    ratios = {acc: round(summary.loc[acc, "ratio"], 4) if acc in summary.index else 0.0
              for acc in all_accounts}
    return ratios


def compute_fan_scores(G: nx.DiGraph) -> dict:
    """
    Fan-in/fan-out asymmetry score.
    Mule: high in-degree from strangers, low out-degree to few accounts.
    Score = in_degree / (out_degree + 1) — capped and normalized.
    """
    scores = {}
    for node in G.nodes():
        in_d  = G.in_degree(node)
        out_d = G.out_degree(node)
        scores[node] = min(in_d / (out_d + 1), 10) / 10   # normalize to [0,1]
    return scores


# ── Composite Mule Score ──────────────────────────────────────────────────────

def compute_mule_scores(
    G: nx.DiGraph,
    df: pd.DataFrame,
    bc: dict,
    pass_through: dict,
    fan: dict,
    govt_flagged: set,
) -> pd.DataFrame:
    """
    Weighted composite mule score per account:
      30% betweenness centrality
      35% pass-through ratio
      20% fan-in/fan-out score
      15% govt alert flag
    """
    accounts = list(G.nodes())

    # Normalize betweenness to [0,1]
    bc_vals = np.array([bc.get(a, 0) for a in accounts])
    bc_norm = (bc_vals - bc_vals.min()) / (bc_vals.max() - bc_vals.min() + 1e-9)

    rows = []
    for i, acc in enumerate(accounts):
        pt    = pass_through.get(acc, 0.0)
        fn    = fan.get(acc, 0.0)
        gov   = 1.0 if acc in govt_flagged else 0.0
        bc_n  = bc_norm[i]

        score = (0.30 * bc_n) + (0.35 * pt) + (0.20 * fn) + (0.15 * gov)
        score = round(min(score, 1.0), 4)

        label = (
            "HIGH_RISK" if score >= MULE_THRESHOLD_HIGH_RISK else
            "SUSPECT"   if score >= MULE_THRESHOLD_SUSPECT   else
            "CLEAN"
        )

        total_in  = sum(G[u][acc]["total_amount"] for u in G.predecessors(acc)) if G.in_degree(acc) > 0 else 0
        total_out = sum(G[acc][v]["total_amount"] for v in G.successors(acc))   if G.out_degree(acc) > 0 else 0

        rows.append({
            "account_id":        acc,
            "mule_score":        score,
            "mule_label":        label,
            "betweenness":       round(bc.get(acc, 0), 6),
            "pass_through_ratio":round(pt, 4),
            "fan_score":         round(fn, 4),
            "govt_flagged":      int(gov),
            "in_degree":         G.in_degree(acc),
            "out_degree":        G.out_degree(acc),
            "total_received":    round(total_in, 2),
            "total_sent":        round(total_out, 2),
        })

    return pd.DataFrame(rows).sort_values("mule_score", ascending=False).reset_index(drop=True)


def get_account_neighborhood(G: nx.DiGraph, account_id: str, depth: int = 2) -> nx.DiGraph:
    """Extract the subgraph around an account for case investigation view."""
    nodes = {account_id}
    for _ in range(depth):
        neighbors = set()
        for n in nodes:
            neighbors.update(G.predecessors(n))
            neighbors.update(G.successors(n))
        nodes.update(neighbors)
    return G.subgraph(nodes).copy()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_graph_detection(df: pd.DataFrame = None, save: bool = True) -> tuple:
    if df is None:
        path = os.path.join(PROCESSED_DIR, "scored_transactions.csv")
        df   = pd.read_csv(path, parse_dates=["timestamp"])

    print("=" * 55)
    print("  FraudShield — Graph-Based Mule Detection")
    print("=" * 55)

    # Build graph incrementally (simulate streaming by processing in chunks)
    tg = TransactionGraph()
    chunk_size = 2000
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        tg.incremental_update(chunk)

    G = tg.G

    # Govt-flagged accounts
    govt_flagged = set(
        df[df["govt_alert"] == 1]["sender_id"].tolist() +
        df[df["govt_alert"] == 1]["receiver_id"].tolist()
    )

    # Compute all signals
    bc           = compute_betweenness(G)
    pass_through = compute_pass_through_ratio(df)
    fan          = compute_fan_scores(G)

    # Composite mule scores
    mule_df = compute_mule_scores(G, df, bc, pass_through, fan, govt_flagged)

    # Summary
    n_high   = (mule_df["mule_label"] == "HIGH_RISK").sum()
    n_suspect= (mule_df["mule_label"] == "SUSPECT").sum()
    n_clean  = (mule_df["mule_label"] == "CLEAN").sum()

    print(f"\n[Results]")
    print(f"  Accounts analyzed  : {len(mule_df):,}")
    print(f"  HIGH_RISK          : {n_high:,}")
    print(f"  SUSPECT            : {n_suspect:,}")
    print(f"  CLEAN              : {n_clean:,}")

    if save:
        out = os.path.join(PROCESSED_DIR, "mule_scores.csv")
        mule_df.to_csv(out, index=False)
        print(f"\n[Saved]  → {out}")

    print("=" * 55)
    return mule_df, G


if __name__ == "__main__":
    mule_df, G = run_graph_detection()

    print("\nTop 10 highest-risk mule accounts:")
    print(mule_df[[
        "account_id", "mule_score", "mule_label",
        "pass_through_ratio", "betweenness", "in_degree", "out_degree", "govt_flagged"
    ]].head(10).to_string(index=False))

    # Demo: neighborhood of top suspect
    top_acc = mule_df.iloc[0]["account_id"]
    subG    = get_account_neighborhood(G, top_acc, depth=1)
    print(f"\nNeighborhood of {top_acc}: {subG.number_of_nodes()} accounts, {subG.number_of_edges()} edges")
