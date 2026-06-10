"""
model_comparison.py
FraudShield — Model Comparison

Trains and evaluates three models side by side:
  1. Logistic Regression  — simple linear baseline
  2. Random Forest        — ensemble baseline
  3. XGBoost              — our chosen model

Metrics: AUC-ROC, AUC-PR, Precision, Recall, F1
Outputs a comparison CSV and prints a summary table.

Run: python detection/model_comparison.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
FEATURES = [
    "amount", "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "tx_count_24h", "amount_zscore", "govt_alert",
    "tx_type_enc", "source_enc", "alert_severity_enc",
]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple:
    path = os.path.join(PROCESSED_DIR, "unified_transactions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run run_pipeline.py first to generate unified_transactions.csv")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Reuse preprocessing from detection module
    from detection.model import preprocess
    df_proc, _, _, _ = preprocess(df)

    X = df_proc[FEATURES].fillna(0)
    y = df_proc["is_fraud"]
    return X, y


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_models(scale_pos_weight: float) -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_train, X_val, y_train, y_val) -> dict:
    """Train and evaluate a single model. Returns metrics dict."""
    print(f"  Training {name}...")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= 0.40).astype(int)

    auc_roc = roc_auc_score(y_val, y_proba)
    auc_pr  = average_precision_score(y_val, y_proba)
    prec    = precision_score(y_val, y_pred, zero_division=0)
    rec     = recall_score(y_val, y_pred, zero_division=0)
    f1      = f1_score(y_val, y_pred, zero_division=0)

    return {
        "Model":          name,
        "AUC-ROC":        round(auc_roc, 4),
        "AUC-PR":         round(auc_pr, 4),
        "Precision":      round(prec, 4),
        "Recall":         round(rec, 4),
        "F1 Score":       round(f1, 4),
        "Val Fraud Count":int(y_val.sum()),
        "Flagged Count":  int(y_pred.sum()),
    }


def cross_validate_model(name: str, model, X, y, cv: int = 5) -> dict:
    """5-fold stratified cross-validation AUC-ROC."""
    print(f"  Cross-validating {name} ({cv}-fold)...")
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return {
        "Model":       name,
        "CV AUC Mean": round(scores.mean(), 4),
        "CV AUC Std":  round(scores.std(), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_comparison(save: bool = True) -> pd.DataFrame:
    print("=" * 60)
    print("  FraudShield — Model Comparison")
    print("=" * 60)

    X, y = load_data()
    print(f"\n[Data]  {len(X):,} transactions | Fraud rate: {y.mean()*100:.2f}%")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = get_models(scale_pos_weight)

    # ── Hold-out evaluation ───────────────────────────────────────────────────
    print(f"\n[Hold-out Evaluation] (80/20 stratified split)")
    holdout_results = []
    for name, model in models.items():
        result = evaluate_model(name, model, X_train, X_val, y_train, y_val)
        holdout_results.append(result)

    holdout_df = pd.DataFrame(holdout_results)

    # ── Cross-validation ──────────────────────────────────────────────────────
    print(f"\n[5-Fold Cross-Validation]")
    cv_results = []
    for name, model in models.items():
        result = cross_validate_model(name, model, X, y, cv=5)
        cv_results.append(result)

    cv_df = pd.DataFrame(cv_results)

    # ── Merge results ─────────────────────────────────────────────────────────
    comparison_df = holdout_df.merge(cv_df, on="Model")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(comparison_df[[
        "Model", "AUC-ROC", "AUC-PR", "Precision", "Recall", "F1 Score",
        "CV AUC Mean", "CV AUC Std"
    ]].to_string(index=False))

    # Winner
    best = comparison_df.loc[comparison_df["AUC-ROC"].idxmax(), "Model"]
    best_auc = comparison_df["AUC-ROC"].max()
    print(f"\n✅ Best model: {best} (AUC-ROC: {best_auc:.4f})")

    # XGBoost vs baseline lift
    xgb_auc = comparison_df[comparison_df["Model"] == "XGBoost"]["AUC-ROC"].values[0]
    lr_auc  = comparison_df[comparison_df["Model"] == "Logistic Regression"]["AUC-ROC"].values[0]
    rf_auc  = comparison_df[comparison_df["Model"] == "Random Forest"]["AUC-ROC"].values[0]
    print(f"   XGBoost vs Logistic Regression: +{(xgb_auc - lr_auc)*100:.2f}% AUC-ROC")
    print(f"   XGBoost vs Random Forest:       +{(xgb_auc - rf_auc)*100:.2f}% AUC-ROC")

    if save:
        out = os.path.join(PROCESSED_DIR, "model_comparison.csv")
        comparison_df.to_csv(out, index=False)
        print(f"\n[Saved]  → {out}")

    print("=" * 60)
    return comparison_df


if __name__ == "__main__":
    run_comparison()
