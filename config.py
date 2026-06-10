"""
config.py
FraudShield — Central Configuration

All paths, thresholds, and settings in one place.
Override any value via environment variables.
"""

import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODEL_DIR      = DATA_DIR / "models"

# Auto-create directories
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data file paths ───────────────────────────────────────────────────────────
IEEE_CIS_PATH   = Path(os.getenv("IEEE_CIS_PATH",   str(RAW_DIR / "ieee_cis_transactions.csv")))
PAYSIM_PATH     = Path(os.getenv("PAYSIM_PATH",     str(RAW_DIR / "paysim_transactions.csv")))
ALERTS_PATH     = Path(os.getenv("ALERTS_PATH",     str(RAW_DIR / "govt_fraud_alerts.json")))
REAL_DATA_PATH  = Path(os.getenv("REAL_DATA_PATH",  str(RAW_DIR / "DataSet.csv")))

UNIFIED_TX_PATH     = PROCESSED_DIR / "unified_transactions.csv"
SCORED_TX_PATH      = PROCESSED_DIR / "scored_transactions.csv"
MULE_SCORES_PATH    = PROCESSED_DIR / "mule_scores.csv"
REAL_PROCESSED_PATH = PROCESSED_DIR / "real_dataset_processed.csv"
REAL_FEATURES_PATH  = PROCESSED_DIR / "real_dataset_features.csv"

ISO_MODEL_PATH  = MODEL_DIR / "isolation_forest.pkl"
XGB_MODEL_PATH  = MODEL_DIR / "xgboost.pkl"

# ── Detection thresholds ──────────────────────────────────────────────────────
RISK_THRESHOLD          = float(os.getenv("RISK_THRESHOLD",           "0.40"))
MULE_THRESHOLD_SUSPECT  = float(os.getenv("MULE_THRESHOLD_SUSPECT",   "0.40"))
MULE_THRESHOLD_HIGH     = float(os.getenv("MULE_THRESHOLD_HIGH_RISK", "0.65"))
PASS_THROUGH_WINDOW_HRS = int(os.getenv("PASS_THROUGH_WINDOW_HRS",    "2"))

# ── ML settings ───────────────────────────────────────────────────────────────
ISO_CONTAMINATION   = float(os.getenv("ISO_CONTAMINATION",   "0.04"))
ISO_N_ESTIMATORS    = int(os.getenv("ISO_N_ESTIMATORS",       "200"))
XGB_N_ESTIMATORS    = int(os.getenv("XGB_N_ESTIMATORS",       "300"))
XGB_MAX_DEPTH       = int(os.getenv("XGB_MAX_DEPTH",           "6"))
XGB_LEARNING_RATE   = float(os.getenv("XGB_LEARNING_RATE",    "0.05"))
TEST_SIZE           = float(os.getenv("TEST_SIZE",             "0.20"))

# ── LLM settings ─────────────────────────────────────────────────────────────
# API key loaded from env var — never hard-coded
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TIMEOUT_SECS  = int(os.getenv("LLM_TIMEOUT_SECS", "30"))

# ── Graph settings ────────────────────────────────────────────────────────────
BETWEENNESS_SAMPLE_K = int(os.getenv("BETWEENNESS_SAMPLE_K", "500"))
GRAPH_CHUNK_SIZE     = int(os.getenv("GRAPH_CHUNK_SIZE",      "2000"))
