# 🛡️ FraudShield

**AI/ML-Powered Fraud Detection & Mule Account Identification System**

Built for Hackathon Problem Statement 2 — detecting suspicious financial transactions and mule accounts using machine learning, graph analysis, and LLM-powered explainability.

---

## Architecture

```
Data Sources → Ingestion Pipeline → Detection Engine → LLM Explainability → Dashboard
```

| Layer | Component | Tech |
|---|---|---|
| Ingestion | Multi-source pipeline | Pandas, JSON |
| Detection | Anomaly scoring | Isolation Forest + XGBoost |
| Graph | Mule account detection | NetworkX (incremental) |
| Explainability | Risk narratives | Claude API |
| Dashboard | 4-page UI | Streamlit + Plotly |

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
# Windows
set ANTHROPIC_API_KEY=your_key_here

# Mac/Linux
export ANTHROPIC_API_KEY=your_key_here
```
Or copy `.env.example` to `.env` and fill in your key.

### 3. Run the pipeline
```bash
# Using synthetic data
python run_pipeline.py

# Using the hackathon dataset
python ingestion/load_real_dataset.py --input path/to/DataSet.csv
python run_pipeline.py
```

### 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
fraudshield/
├── config.py                     ← Central config (paths, thresholds, settings)
├── run_pipeline.py               ← Full pipeline bootstrap
├── requirements.txt
├── .env.example                  ← Copy to .env for secret management
│
├── ingestion/
│   ├── generate_sample_data.py   ← Synthetic IEEE-CIS, PaySim, Govt alerts
│   ├── pipeline.py               ← Unified schema + feature engineering
│   └── load_real_dataset.py      ← Hackathon dataset loader
│
├── detection/
│   └── model.py                  ← Isolation Forest + XGBoost + composite score
│
├── graph/
│   └── graph_detector.py         ← Incremental graph + mule scoring
│
├── llm/
│   └── explainer.py              ← Claude API risk narratives
│
├── dashboard/
│   └── app.py                    ← Streamlit dashboard (4 pages)
│
├── tests/
│   └── test_fraudshield.py       ← Automated test suite (pytest)
│
└── data/
    ├── raw/                      ← Input CSVs and JSON
    ├── processed/                ← Scored transactions, mule scores
    └── models/                   ← Trained .pkl files
```

---

## Detection Methodology

### Transaction-Level (ML)
- **Isolation Forest** — unsupervised anomaly scoring on all transactions
- **XGBoost** — supervised fraud classifier with class imbalance handling
- **Composite score** — 40% ISO + 50% XGB + 10% govt alert boost

### Account-Level (Graph)
- Incremental directed transaction graph (NetworkX)
- Mule signals: betweenness centrality, pass-through ratio, fan-in/out asymmetry, govt alert flags
- Labels: `CLEAN` / `SUSPECT` / `HIGH_RISK`

### Explainability (LLM)
- Flagged transactions and accounts passed to Claude API
- Generates auditor-ready risk narratives with verdict: `LOW / MEDIUM / HIGH / CRITICAL`
- Feedback loop: confirmed fraud retrains and reweights models

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Datasets

| Dataset | Source | Usage |
|---|---|---|
| IEEE-CIS Fraud Detection | Kaggle | Labeled transaction data |
| PaySim Synthetic Transactions | Kaggle | Mobile money with fraud labels |
| Govt Cyber Fraud Alerts | Mock JSON | Regulatory alert cross-reference |
| Hackathon Dataset | Provided | 9,082 accounts, 3,924 anonymized features |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for LLM explanations |
| `RISK_THRESHOLD` | `0.40` | ML flag threshold |
| `MULE_THRESHOLD_SUSPECT` | `0.40` | Mule suspect threshold |
| `MULE_THRESHOLD_HIGH_RISK` | `0.65` | Mule high-risk threshold |
| `IEEE_CIS_PATH` | `data/raw/ieee_cis_transactions.csv` | Override data path |
| `PAYSIM_PATH` | `data/raw/paysim_transactions.csv` | Override data path |
| `REAL_DATA_PATH` | `data/raw/DataSet.csv` | Override hackathon dataset path |
