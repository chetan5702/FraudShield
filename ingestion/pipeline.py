"""
pipeline.py
FraudShield — Data Ingestion & Normalization Pipeline

Loads IEEE-CIS, PaySim, and Govt Alert data, normalizes them into a
unified transaction schema, cross-references alerts, and outputs a
clean DataFrame ready for the detection engine.

Unified Schema:
    transaction_id  | str   — unique ID
    timestamp       | datetime
    sender_id       | str   — source account
    receiver_id     | str   — destination account
    amount          | float — transaction amount (INR equivalent)
    tx_type         | str   — PAYMENT / TRANSFER / CASH_OUT / DEBIT / CASH_IN / CARD
    source          | str   — 'ieee_cis' | 'paysim'
    is_fraud        | int   — ground truth label (0/1)
    govt_alert      | int   — 1 if sender/receiver flagged by govt alert
    alert_severity  | str   — severity of matched alert (or 'NONE')
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

RAW_DIR       = os.path.join(os.path.dirname(__file__), "../data/raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SEVERITY_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_ieee_cis(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["TransactionDT"])
    unified = pd.DataFrame({
        "transaction_id": df["TransactionID"],
        "timestamp":      pd.to_datetime(df["TransactionDT"]),
        "sender_id":      df["sender_id"],
        "receiver_id":    df["receiver_id"],
        "amount":         df["TransactionAmt"].clip(lower=0),
        "tx_type":        "CARD",
        "source":         "ieee_cis",
        "is_fraud":       df["isFraud"].astype(int),
    })
    print(f"[IEEE-CIS]  Loaded {len(unified):,} transactions")
    return unified


def load_paysim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    unified = pd.DataFrame({
        "transaction_id": "TXN_PS_" + df.index.astype(str).str.zfill(6),
        "timestamp":      pd.to_datetime(df["timestamp"]),
        "sender_id":      df["nameOrig"],
        "receiver_id":    df["nameDest"],
        "amount":         df["amount"].clip(lower=0),
        "tx_type":        df["type"],
        "source":         "paysim",
        "is_fraud":       df["isFraud"].astype(int),
    })
    print(f"[PaySim]    Loaded {len(unified):,} transactions")
    return unified


def load_govt_alerts(path: str) -> pd.DataFrame:
    with open(path) as f:
        alerts = json.load(f)
    df = pd.DataFrame(alerts)
    df["issued_at"] = pd.to_datetime(df["issued_at"])
    print(f"[Govt]      Loaded {len(df):,} alerts  "
          f"({df['severity'].value_counts().to_dict()})")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]     = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many transactions each sender made in the past 24 hours."""
    df = df.sort_values("timestamp").copy()

    # True 24-hour rolling window: for each tx, count prior txns by the same
    # sender within the preceding 24 hours (exclusive of the current row).
    # Strategy: use searchsorted on per-sender timestamp arrays.
    cutoff_ns = pd.Timedelta(hours=24).value  # nanoseconds

    counts = np.empty(len(df), dtype=np.int64)
    for sender, grp in df.groupby("sender_id"):
        ts_ns = grp["timestamp"].values.astype(np.int64)  # nanoseconds
        idx   = grp.index
        # For each position i, count entries in ts_ns within (ts_ns[i]-24h, ts_ns[i])
        lo = np.searchsorted(ts_ns, ts_ns - cutoff_ns, side="left")
        hi = np.arange(len(ts_ns))          # excludes current row (strict past)
        counts[df.index.get_indexer(idx)] = np.maximum(hi - lo, 0)

    df["tx_count_24h"] = counts
    # Amount z-score per sender (deviation from their own avg)
    sender_stats = df.groupby("sender_id")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "sender_mean_amt", "std": "sender_std_amt"}
    )
    df = df.join(sender_stats, on="sender_id")
    df["sender_std_amt"]   = df["sender_std_amt"].fillna(1).replace(0, 1)
    df["amount_zscore"]    = (df["amount"] - df["sender_mean_amt"]) / df["sender_std_amt"]
    return df


def cross_reference_alerts(df: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """Flag transactions where sender or receiver appears in govt alert list."""
    # Keep only unresolved or critical alerts
    active = alerts[~alerts["resolved"] | (alerts["severity"] == "CRITICAL")]

    # Build account → worst severity mapping
    severity_map = (
        active.groupby("account_id")["severity"]
        .apply(lambda s: max(s, key=lambda x: SEVERITY_RANK[x]))
        .to_dict()
    )

    flagged_accounts = set(severity_map.keys())

    df = df.copy()
    df["govt_alert"] = (
        df["sender_id"].isin(flagged_accounts) |
        df["receiver_id"].isin(flagged_accounts)
    ).astype(int)

    df["alert_severity"] = df.apply(
        lambda r: max(
            severity_map.get(r["sender_id"], "NONE"),
            severity_map.get(r["receiver_id"], "NONE"),
            key=lambda x: SEVERITY_RANK[x]
        ), axis=1
    )
    n_flagged = df["govt_alert"].sum()
    print(f"[Alerts]    {n_flagged:,} transactions cross-referenced with govt alerts")
    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    ieee_path:  str = None,
    paysim_path: str = None,
    alerts_path: str = None,
    save: bool = True,
) -> pd.DataFrame:

    ieee_path   = ieee_path   or os.path.join(RAW_DIR, "ieee_cis_transactions.csv")
    paysim_path = paysim_path or os.path.join(RAW_DIR, "paysim_transactions.csv")
    alerts_path = alerts_path or os.path.join(RAW_DIR, "govt_fraud_alerts.json")

    print("=" * 55)
    print("  FraudShield — Data Ingestion Pipeline")
    print("=" * 55)

    # Load
    ieee_df   = load_ieee_cis(ieee_path)
    paysim_df = load_paysim(paysim_path)
    alerts_df = load_govt_alerts(alerts_path)

    # Merge into unified schema
    df = pd.concat([ieee_df, paysim_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"\n[Merged]    {len(df):,} total transactions")

    # Feature engineering
    df = add_time_features(df)
    df = add_velocity_features(df)
    df = cross_reference_alerts(df, alerts_df)

    # Summary
    fraud_rate = df["is_fraud"].mean() * 100
    govt_rate  = df["govt_alert"].mean() * 100
    print(f"\n[Summary]")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Fraud rate         : {fraud_rate:.2f}%")
    print(f"  Govt-flagged       : {govt_rate:.2f}%")
    print(f"  Date range         : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Unique accounts    : {pd.concat([df['sender_id'], df['receiver_id']]).nunique():,}")

    if save:
        out_path = os.path.join(PROCESSED_DIR, "unified_transactions.csv")
        df.to_csv(out_path, index=False)
        print(f"\n[Saved]     → {out_path}")

    print("=" * 55)
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print("\nSample output:")
    print(df[["transaction_id", "sender_id", "receiver_id", "amount",
              "tx_type", "is_fraud", "govt_alert", "alert_severity"]].head(10).to_string(index=False))
