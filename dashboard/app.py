"""
app.py
FraudShield — Main Dashboard

Run with: streamlit run dashboard/app.py

Pages:
  1. Overview      — KPI cards + risk distribution charts
  2. Transactions  — flagged transaction table with LLM explanation drill-down
  3. Mule Accounts — graph-scored accounts with case investigation view
  4. Graph View    — interactive account network visualization
"""

import os
import sys
import json
import pickle
import requests
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PROCESSED  = os.path.join(ROOT, "data/processed")
MODEL_DIR  = os.path.join(ROOT, "data/models")

# ── Load tuned threshold (falls back to 0.40 if models haven't been trained yet) ──
def _load_threshold() -> float:
    path = os.path.join(MODEL_DIR, "threshold.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return 0.40

RISK_THRESHOLD = _load_threshold()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e1e2e; border-radius: 10px; padding: 20px;
    border-left: 4px solid #e8a020; color: white; margin-bottom: 10px;
  }
  .metric-label { font-size: 13px; color: #aaa; margin-bottom: 4px; }
  .metric-value { font-size: 32px; font-weight: bold; color: #fff; }
  .metric-sub   { font-size: 12px; color: #e8a020; margin-top: 4px; }
  .verdict-CRITICAL { color: #ff4444; font-weight: bold; }
  .verdict-HIGH     { color: #ff8800; font-weight: bold; }
  .verdict-MEDIUM   { color: #ffcc00; font-weight: bold; }
  .verdict-LOW      { color: #44cc44; font-weight: bold; }
  .explanation-box  {
    background: #1e1e2e; border-radius: 8px; padding: 16px;
    border-left: 3px solid #e8a020; color: #eee; font-size: 14px; line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    tx   = pd.read_csv(os.path.join(PROCESSED, "scored_transactions.csv"), parse_dates=["timestamp"])
    mule = pd.read_csv(os.path.join(PROCESSED, "mule_scores.csv"))
    return tx, mule


@st.cache_resource
def build_graph(tx: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    # Aggregate edge weights before building the graph — avoids Python-level
    # row-by-row iteration which is O(n) Python overhead per row.
    edge_agg = (
        tx.groupby(["sender_id", "receiver_id"])["amount"]
        .agg(weight="sum", count="count")
        .reset_index()
    )
    for row in edge_agg.itertuples(index=False):
        G.add_edge(row.sender_id, row.receiver_id, weight=row.weight, count=row.count)
    return G


# ── LLM explainer ─────────────────────────────────────────────────────────────
def get_llm_explanation(prompt: str, gemini_key: str = "", claude_key: str = "") -> dict:
    try:
        from llm.explainer import call_llm
        return call_llm(prompt, gemini_key=gemini_key or None, claude_key=claude_key or None)
    except Exception as e:
        return {"explanation": f"Explanation unavailable: {e}", "verdict": "UNKNOWN"}


def txn_prompt(row, mule_lookup):
    sm = mule_lookup.get(row["sender_id"], {})
    rm = mule_lookup.get(row["receiver_id"], {})
    return f"""
Transaction flagged by FraudShield:
- ID: {row['transaction_id']} | Amount: ₹{row['amount']:,.2f} | Type: {row['tx_type']}
- Timestamp: {row['timestamp']} | Night: {row.get('is_night',0)} | Weekend: {row.get('is_weekend',0)}
- Sender: {row['sender_id']} | Receiver: {row['receiver_id']}
- Amount Z-score: {row.get('amount_zscore', 0):.2f} | TX count 24h: {row.get('tx_count_24h', 0)}
- Govt Alert: {'Yes' if row.get('govt_alert') else 'No'} ({row.get('alert_severity','NONE')})
- Risk Score: {row.get('risk_score',0):.3f} | XGBoost: {row.get('xgb_proba',0):.3f}
- Sender mule label: {sm.get('mule_label','N/A')} (score {sm.get('mule_score','N/A')})
- Receiver mule label: {rm.get('mule_label','N/A')} (score {rm.get('mule_score','N/A')})
Explain why this was flagged. Return JSON with explanation and verdict.
"""


def acc_prompt(row, tx_df):
    acc  = row["account_id"]
    txns = tx_df[(tx_df["sender_id"] == acc) | (tx_df["receiver_id"] == acc)]
    return f"""
Account suspected as mule by FraudShield:
- Account: {acc} | Mule Score: {row['mule_score']} | Label: {row['mule_label']}
- Betweenness: {row['betweenness']:.6f} | Pass-through ratio: {row['pass_through_ratio']:.4f}
- Fan score: {row['fan_score']:.4f} | Govt flagged: {'Yes' if row['govt_flagged'] else 'No'}
- In-degree: {row['in_degree']} | Out-degree: {row['out_degree']}
- Total received: ₹{row['total_received']:,.2f} | Total sent: ₹{row['total_sent']:,.2f}
- Total transactions: {len(txns)}
Explain why this account is suspected as a mule. Return JSON with explanation and verdict.
"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("FraudShield")
    st.caption("AI-Powered Fraud Detection")
    st.divider()

    page = st.radio("Navigation", ["📊 Overview", "🚨 Flagged Transactions", "🕵️ Mule Accounts", "🕸️ Graph View", "🏆 Model Comparison"])
    st.divider()

    st.divider()
    st.caption("LLM API Key (for explanations)")
    gemini_key = st.text_input("Gemini API Key (free)", type="password",
                                help="Get free key at aistudio.google.com")
    api_key    = st.text_input("Anthropic API Key (optional)", type="password",
                                help="Fallback if Gemini key not provided")
    st.caption("Keys used only for live API calls. Not stored.")


# ── Load data ─────────────────────────────────────────────────────────────────
tx_df, mule_df = load_data()
G = build_graph(tx_df)
mule_lookup = mule_df.set_index("account_id").to_dict(orient="index")

flagged = tx_df[tx_df["ml_flagged"] == 1].sort_values("risk_score", ascending=False)
suspects = mule_df[mule_df["mule_label"].isin(["SUSPECT", "HIGH_RISK"])]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🛡️ FraudShield — Overview")
    st.caption(f"Last updated: {tx_df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">{len(tx_df):,}</div>
            <div class="metric-sub">Across all sources</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ML Flagged</div>
            <div class="metric-value">{len(flagged):,}</div>
            <div class="metric-sub">{len(flagged)/len(tx_df)*100:.1f}% flag rate</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Suspect Accounts</div>
            <div class="metric-value">{len(suspects):,}</div>
            <div class="metric-sub">{(mule_df['mule_label']=='HIGH_RISK').sum()} HIGH_RISK</div></div>""", unsafe_allow_html=True)
    with c4:
        govt_flagged_count = tx_df["govt_alert"].sum()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Govt-Flagged Txns</div>
            <div class="metric-value">{govt_flagged_count:,}</div>
            <div class="metric-sub">Cross-referenced alerts</div></div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(tx_df, x="risk_score", nbins=50, color_discrete_sequence=["#e8a020"],
                           labels={"risk_score": "Risk Score", "count": "Transactions"})
        fig.add_vline(x=RISK_THRESHOLD, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold ({RISK_THRESHOLD:.3f})")
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Mule Account Labels")
        counts = mule_df["mule_label"].value_counts().reset_index()
        fig = px.pie(counts, names="mule_label", values="count",
                     color_discrete_map={"CLEAN": "#44cc44", "SUSPECT": "#ffcc00", "HIGH_RISK": "#ff4444"})
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Flagged Transactions by Hour")
        hourly = flagged.groupby("hour_of_day").size().reset_index(name="count")
        fig = px.bar(hourly, x="hour_of_day", y="count", color_discrete_sequence=["#e8a020"])
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Risk Score by Transaction Type")
        fig = px.box(tx_df, x="tx_type", y="risk_score", color_discrete_sequence=["#e8a020"])
        fig.add_hline(y=RISK_THRESHOLD, line_dash="dash", line_color="red")
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Flagged Transactions
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Flagged Transactions":
    st.title("🚨 Flagged Transactions")
    st.caption(f"{len(flagged):,} transactions above risk threshold")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Min Risk Score", 0.0, 1.0, float(RISK_THRESHOLD), 0.05)
    with col2:
        tx_types = ["All"] + sorted(flagged["tx_type"].unique().tolist())
        sel_type = st.selectbox("Transaction Type", tx_types)
    with col3:
        govt_only = st.checkbox("Govt-Flagged Only")

    filtered = flagged[flagged["risk_score"] >= min_score]
    if sel_type != "All":
        filtered = filtered[filtered["tx_type"] == sel_type]
    if govt_only:
        filtered = filtered[filtered["govt_alert"] == 1]

    st.caption(f"Showing {len(filtered):,} transactions")

    # Table
    display_cols = ["transaction_id", "sender_id", "receiver_id", "amount",
                    "tx_type", "risk_score", "xgb_proba", "govt_alert", "alert_severity"]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "transaction_id": "ID", "sender_id": "Sender", "receiver_id": "Receiver",
            "amount": "Amount (₹)", "tx_type": "Type", "risk_score": "Risk Score",
            "xgb_proba": "XGB Prob", "govt_alert": "Govt Flag", "alert_severity": "Severity"
        }),
        use_container_width=True, height=300
    )

    # Drill-down
    st.divider()
    st.subheader("🔍 Case Investigation")
    sel_txn = st.selectbox("Select transaction to investigate", filtered["transaction_id"].tolist()[:50])

    if sel_txn:
        row = filtered[filtered["transaction_id"] == sel_txn].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Amount", f"₹{row['amount']:,.2f}")
        c2.metric("Risk Score", f"{row['risk_score']:.3f}")
        c3.metric("Govt Alert", "Yes ⚠️" if row['govt_alert'] else "No ✅")

        if gemini_key or api_key:
            with st.spinner("Generating LLM risk narrative..."):
                result = get_llm_explanation(txn_prompt(row, mule_lookup), gemini_key=gemini_key, claude_key=api_key)
            verdict = result.get("verdict", "UNKNOWN")
            st.markdown(f"**Verdict:** <span class='verdict-{verdict}'>{verdict}</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='explanation-box'>{result.get('explanation','')}</div>", unsafe_allow_html=True)
        else:
            st.info("Enter your Gemini API key (free) or Anthropic API key in the sidebar to generate LLM risk narratives.")

        # Show sender/receiver mule profiles
        st.divider()
        mc1, mc2 = st.columns(2)
        with mc1:
            st.subheader(f"Sender: {row['sender_id']}")
            sm = mule_lookup.get(row["sender_id"], {})
            if sm:
                st.metric("Mule Score", sm.get("mule_score", "N/A"))
                st.metric("Label", sm.get("mule_label", "N/A"))
                st.metric("Pass-through Ratio", sm.get("pass_through_ratio", "N/A"))
        with mc2:
            st.subheader(f"Receiver: {row['receiver_id']}")
            rm = mule_lookup.get(row["receiver_id"], {})
            if rm:
                st.metric("Mule Score", rm.get("mule_score", "N/A"))
                st.metric("Label", rm.get("mule_label", "N/A"))
                st.metric("Pass-through Ratio", rm.get("pass_through_ratio", "N/A"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Mule Accounts
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🕵️ Mule Accounts":
    st.title("🕵️ Mule Account Investigation")

    label_filter = st.multiselect("Filter by label", ["HIGH_RISK", "SUSPECT", "CLEAN"],
                                   default=["HIGH_RISK", "SUSPECT"])
    filtered_mule = mule_df[mule_df["mule_label"].isin(label_filter)].sort_values("mule_score", ascending=False)

    st.dataframe(filtered_mule, use_container_width=True, height=300)

    st.divider()
    st.subheader("🔍 Account Deep Dive")
    sel_acc = st.selectbox("Select account", filtered_mule["account_id"].tolist()[:50])

    if sel_acc:
        row = filtered_mule[filtered_mule["account_id"] == sel_acc].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mule Score", f"{row['mule_score']:.4f}")
        c2.metric("Pass-through", f"{row['pass_through_ratio']:.2%}")
        c3.metric("In-degree", row["in_degree"])
        c4.metric("Out-degree", row["out_degree"])

        # LLM explanation
        if gemini_key or api_key:
            with st.spinner("Generating LLM account risk narrative..."):
                result = get_llm_explanation(acc_prompt(row, tx_df), gemini_key=gemini_key, claude_key=api_key)
            verdict = result.get("verdict", "UNKNOWN")
            st.markdown(f"**Verdict:** <span class='verdict-{verdict}'>{verdict}</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='explanation-box'>{result.get('explanation','')}</div>", unsafe_allow_html=True)
        else:
            st.info("Enter your Gemini API key (free) or Anthropic API key in the sidebar for LLM risk narratives.")

        # Account transaction history
        st.divider()
        st.subheader("Transaction History")
        acc_txns = tx_df[(tx_df["sender_id"] == sel_acc) | (tx_df["receiver_id"] == sel_acc)]
        acc_txns = acc_txns.sort_values("timestamp", ascending=False)
        st.dataframe(acc_txns[["transaction_id", "timestamp", "sender_id", "receiver_id",
                                "amount", "tx_type", "risk_score", "ml_flagged"]].head(50),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Graph View
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🕸️ Graph View":
    st.title("🕸️ Transaction Network Graph")

    acc_input = st.text_input("Enter Account ID to explore neighborhood", value="ACC_00027")
    depth     = st.slider("Graph depth", 1, 3, 1)

    if acc_input and acc_input in G:
        # Extract neighborhood
        nodes = {acc_input}
        for _ in range(depth):
            nbrs = set()
            for n in nodes:
                nbrs.update(G.predecessors(n))
                nbrs.update(G.successors(n))
            nodes.update(nbrs)

        subG = G.subgraph(nodes)
        pos  = nx.spring_layout(subG, seed=42)

        # Build plotly graph
        edge_x, edge_y = [], []
        for u, v in subG.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(width=0.5, color="#555"), hoverinfo="none")

        node_x, node_y, node_text, node_color = [], [], [], []
        for node in subG.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            ml = mule_lookup.get(node, {}).get("mule_label", "CLEAN")
            ms = mule_lookup.get(node, {}).get("mule_score", 0)
            node_text.append(f"{node}<br>Label: {ml}<br>Score: {ms}")
            node_color.append(
                "#ff4444" if ml == "HIGH_RISK" else
                "#ffcc00" if ml == "SUSPECT"   else
                "#44cc44"
            )

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=10, color=node_color, line=dict(width=1, color="#fff")),
            text=[n if n == acc_input else "" for n in subG.nodes()],
            textposition="top center", hovertext=node_text, hoverinfo="text",
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False, hovermode="closest",
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            height=550,
                        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing {subG.number_of_nodes()} accounts, {subG.number_of_edges()} edges | 🟥 HIGH_RISK  🟨 SUSPECT  🟩 CLEAN")
    else:
        st.warning(f"Account '{acc_input}' not found in graph. Try ACC_00027")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison")
    st.caption("Logistic Regression vs Random Forest vs XGBoost")

    comp_path = os.path.join(PROCESSED, "model_comparison.csv")
    if not os.path.exists(comp_path):
        st.warning("Model comparison not run yet. Run: `py detection/model_comparison.py`")
    else:
        comp_df = pd.read_csv(comp_path)

        # KPI row
        best_model = comp_df.loc[comp_df["AUC-ROC"].idxmax()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Model",    best_model["Model"])
        c2.metric("Best AUC-ROC", f"{best_model['AUC-ROC']:.4f}")
        c3.metric("Best F1",      f"{best_model['F1 Score']:.4f}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("AUC-ROC Comparison")
            fig = px.bar(comp_df, x="Model", y="AUC-ROC",
                         color="Model",
                         color_discrete_sequence=["#534AB7", "#e8a020", "#ff4444"],
                         text="AUC-ROC")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="white", showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Precision vs Recall")
            fig = px.scatter(comp_df, x="Recall", y="Precision",
                             text="Model", size="F1 Score",
                             color="Model",
                             color_discrete_sequence=["#534AB7", "#e8a020", "#ff4444"])
            fig.update_traces(textposition="top center")
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="white", showlegend=False,
                              xaxis_range=[0, 1], yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("F1 Score Comparison")
            fig = px.bar(comp_df, x="Model", y="F1 Score",
                         color="Model",
                         color_discrete_sequence=["#534AB7", "#e8a020", "#ff4444"],
                         text="F1 Score")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="white", showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Cross-Validation AUC (5-fold)")
            fig = px.bar(comp_df, x="Model", y="CV AUC Mean",
                         error_y="CV AUC Std",
                         color="Model",
                         color_discrete_sequence=["#534AB7", "#e8a020", "#ff4444"],
                         text="CV AUC Mean")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="white", showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Full Metrics Table")
        st.dataframe(comp_df, use_container_width=True)

        st.divider()
        xgb_row = comp_df[comp_df["Model"] == "XGBoost"].iloc[0]
        lr_row  = comp_df[comp_df["Model"] == "Logistic Regression"].iloc[0]
        rf_row  = comp_df[comp_df["Model"] == "Random Forest"].iloc[0]
        st.subheader("Why XGBoost?")
        st.markdown(f"""
XGBoost achieves the best **F1 score ({xgb_row['F1 Score']:.4f})** and **Precision ({xgb_row['Precision']:.4f})**
among all models — critical for fraud detection where false positives are costly.

- vs Logistic Regression: **+{(xgb_row['F1 Score']-lr_row['F1 Score'])*100:.1f}% F1**, showing non-linear patterns matter
- vs Random Forest: **+{(xgb_row['F1 Score']-rf_row['F1 Score'])*100:.1f}% F1**, with better handling of class imbalance via `scale_pos_weight`
- XGBoost also supports **incremental learning** — enabling the feedback loop for continuous retraining
        """)
