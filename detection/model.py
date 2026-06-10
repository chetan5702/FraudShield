"""
model.py
FraudShield — ML Anomaly Detection Engine

Two-pronged detection:
  1. Isolation Forest  — unsupervised anomaly scoring (no labels needed)
  2. XGBoost Classifier — supervised fraud prediction (uses is_fraud labels)

Both scores are combined into a final composite risk score [0–1].
Transactions above the threshold are flagged for review.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "../data/models")
os.makedirs(MODEL_DIR, exist_ok=True)

RISK_THRESHOLD = 0.40   # fallback default; overridden at runtime by tune_threshold()
FEATURES = [
    "amount", "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "tx_count_24h", "amount_zscore", "govt_alert",
    "tx_type_enc", "source_enc", "alert_severity_enc",
]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    le_tx   = LabelEncoder()
    le_src  = LabelEncoder()
    le_sev  = LabelEncoder()

    df["tx_type_enc"]        = le_tx.fit_transform(df["tx_type"].fillna("UNKNOWN"))
    df["source_enc"]         = le_src.fit_transform(df["source"].fillna("unknown"))
    df["alert_severity_enc"] = le_sev.fit_transform(df["alert_severity"].fillna("NONE"))

    # Clip extreme amounts
    df["amount"] = df["amount"].clip(upper=df["amount"].quantile(0.999))

    # Fill any NaNs
    df[FEATURES] = df[FEATURES].fillna(0)

    return df, le_tx, le_src, le_sev


# ── Isolation Forest ──────────────────────────────────────────────────────────

def train_isolation_forest(X: pd.DataFrame) -> IsolationForest:
    print("[Isolation Forest] Training on full dataset (unsupervised)...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.04,   # expected ~4% anomaly rate
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)
    print("[Isolation Forest] Done.")
    return iso


def iso_score(iso: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Returns anomaly probability [0–1]. Higher = more anomalous."""
    raw = iso.decision_function(X)          # higher = more normal
    normalized = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return normalized


# ── XGBoost Classifier ────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    print("[XGBoost] Training supervised classifier...")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # handles class imbalance
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred  = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    auc     = roc_auc_score(y_val, y_proba)

    print(f"[XGBoost] Validation AUC-ROC: {auc:.4f}")
    print("[XGBoost] Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Legit", "Fraud"], digits=3))
    return clf


# ── Composite Risk Score ──────────────────────────────────────────────────────

def composite_score(iso_scores, xgb_probas, govt_alert_flags) -> np.ndarray:
    """
    Weighted combination:
      40% Isolation Forest anomaly score
      50% XGBoost fraud probability
      10% Govt alert boost
    """
    govt_boost = np.array(govt_alert_flags) * 0.10
    score = (0.40 * iso_scores) + (0.50 * xgb_probas) + govt_boost
    return np.clip(score, 0, 1)


def tune_threshold(
    composite_scores_val: np.ndarray,
    y_val: np.ndarray,
    min_recall: float = 0.80,
) -> float:
    """
    Choose the highest-precision threshold on the validation set that still
    achieves at least `min_recall` (default 80%).

    Using the PR curve is the correct operating-point selection method for
    imbalanced fraud data — AUC-ROC can be misleadingly optimistic when the
    negative class dominates.

    Returns the selected threshold (float).
    """
    precision, recall, thresholds = precision_recall_curve(y_val, composite_scores_val)
    # precision_recall_curve appends a final point with no matching threshold
    precision = precision[:-1]
    recall    = recall[:-1]

    # Mask to thresholds where recall >= min_recall
    viable = recall >= min_recall
    if not viable.any():
        # If nothing meets the recall floor, fall back to max-F1 threshold
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        best_idx = np.argmax(f1)
        selected = float(thresholds[best_idx])
        print(f"[Threshold] No threshold meets {min_recall:.0%} recall floor; "
              f"using max-F1 threshold: {selected:.4f}  "
              f"(P={precision[best_idx]:.3f}, R={recall[best_idx]:.3f})")
        return selected

    # Among viable thresholds, pick the one with the highest precision.
    # Using argmax(precision * viable) can silently pick a non-viable index
    # when all viable precisions are 0 (tie at 0 → argmax returns index 0).
    # Instead, index into only the viable positions directly.
    viable_indices = np.where(viable)[0]
    best_idx = viable_indices[np.argmax(precision[viable_indices])]
    selected = float(thresholds[best_idx])
    print(f"[Threshold] PR-curve tuned threshold: {selected:.4f}  "
          f"(P={precision[best_idx]:.3f}, R={recall[best_idx]:.3f}, "
          f"min_recall constraint={min_recall:.0%})")
    return selected




def run_detection(df: pd.DataFrame = None, save_models: bool = True) -> pd.DataFrame:
    if df is None:
        path = os.path.join(PROCESSED_DIR, "unified_transactions.csv")
        df   = pd.read_csv(path, parse_dates=["timestamp"])

    print("=" * 55)
    print("  FraudShield — ML Detection Engine")
    print("=" * 55)

    df_proc, le_tx, le_src, le_sev = preprocess(df)
    X = df_proc[FEATURES]
    y = df_proc["is_fraud"]

    # Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Train models
    iso = train_isolation_forest(X)              # trained on full data (unsupervised)
    clf = train_xgboost(X_train, y_train, X_val, y_val)

    # Score full dataset
    iso_scores  = iso_score(iso, X)
    xgb_probas  = clf.predict_proba(X)[:, 1]
    risk_scores = composite_score(iso_scores, xgb_probas, df_proc["govt_alert"].values)

    # Derive threshold from validation composite scores (not hardcoded)
    iso_scores_val  = iso_score(iso, X_val)
    xgb_probas_val  = clf.predict_proba(X_val)[:, 1]
    composite_val   = composite_score(iso_scores_val, xgb_probas_val,
                                      df_proc.loc[X_val.index, "govt_alert"].values)
    threshold = tune_threshold(composite_val, y_val.values, min_recall=0.80)

    # Attach results back to original df
    result = df.copy()
    result["iso_score"]    = iso_scores
    result["xgb_proba"]    = xgb_probas
    result["risk_score"]   = risk_scores
    result["ml_flagged"]   = (risk_scores >= threshold).astype(int)

    # Stats
    n_flagged  = result["ml_flagged"].sum()
    n_fraud    = result["is_fraud"].sum()
    n_caught   = result[result["ml_flagged"] == 1]["is_fraud"].sum()

    print(f"\n[Results]")
    print(f"  Transactions scored  : {len(result):,}")
    print(f"  ML flagged           : {n_flagged:,}  ({n_flagged/len(result)*100:.1f}%)")
    print(f"  True fraud in data   : {n_fraud:,}")
    print(f"  Fraud caught         : {n_caught:,} / {n_fraud}  ({n_caught/max(n_fraud,1)*100:.1f}% recall)")
    print(f"  Risk threshold       : {threshold:.4f}  (PR-curve tuned, min recall 80%)")

    # Save
    if save_models:
        pickle.dump(iso, open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "wb"))
        pickle.dump(clf, open(os.path.join(MODEL_DIR, "xgboost.pkl"), "wb"))
        pickle.dump(threshold, open(os.path.join(MODEL_DIR, "threshold.pkl"), "wb"))
        result.to_csv(os.path.join(PROCESSED_DIR, "scored_transactions.csv"), index=False)
        print(f"\n[Saved]  Models → data/models/")
        print(f"[Saved]  Threshold ({threshold:.4f}) → data/models/threshold.pkl")
        print(f"[Saved]  Scored data → data/processed/scored_transactions.csv")

    print("=" * 55)
    return result


if __name__ == "__main__":
    result = run_detection()
    print("\nTop 10 highest risk transactions:")
    cols = ["transaction_id", "sender_id", "receiver_id", "amount",
            "tx_type", "risk_score", "ml_flagged", "is_fraud"]
    print(result.sort_values("risk_score", ascending=False)[cols].head(10).to_string(index=False))
