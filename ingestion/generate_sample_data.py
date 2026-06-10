"""
generate_sample_data.py
Generates synthetic transaction data mimicking IEEE-CIS and PaySim datasets.
Run this once to create sample CSVs in data/raw/.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
os.makedirs(RAW_DIR, exist_ok=True)


# ── 1. IEEE-CIS style transactions ──────────────────────────────────────────
def generate_ieee_cis(n=5000):
    account_ids = [f"ACC_{i:05d}" for i in range(500)]
    card_ids    = [f"CARD_{i:04d}" for i in range(300)]
    base_time   = datetime(2024, 1, 1)

    rows = []
    for i in range(n):
        sender   = random.choice(account_ids)
        receiver = random.choice(account_ids)
        while receiver == sender:
            receiver = random.choice(account_ids)

        is_fraud = 1 if random.random() < 0.035 else 0   # ~3.5% fraud rate
        amount   = round(np.random.exponential(200) * (5 if is_fraud else 1), 2)
        amount   = min(amount, 50000)

        rows.append({
            "TransactionID": f"TXN_CIS_{i:06d}",
            "TransactionDT":  (base_time + timedelta(seconds=random.randint(0, 86400*180))).isoformat(),
            "TransactionAmt": amount,
            "card1":          random.choice(card_ids),
            "addr1":          random.randint(100, 500),
            "P_emaildomain":  random.choice(["gmail.com", "yahoo.com", "outlook.com", "protonmail.com"]),
            "sender_id":      sender,
            "receiver_id":    receiver,
            "isFraud":        is_fraud,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(RAW_DIR, "ieee_cis_transactions.csv")
    df.to_csv(path, index=False)
    print(f"[IEEE-CIS]  {len(df):,} rows → {path}")
    return df


# ── 2. PaySim style transactions ─────────────────────────────────────────────
def generate_paysim(n=5000):
    account_ids = [f"ACC_{i:05d}" for i in range(500)]
    tx_types    = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    base_time   = datetime(2024, 1, 1)

    rows = []
    for i in range(n):
        sender   = random.choice(account_ids)
        receiver = random.choice(account_ids)
        while receiver == sender:
            receiver = random.choice(account_ids)

        tx_type  = random.choice(tx_types)
        is_fraud = 1 if (tx_type in ["TRANSFER", "CASH_OUT"] and random.random() < 0.06) else 0
        amount   = round(np.random.exponential(300) * (4 if is_fraud else 1), 2)
        amount   = min(amount, 100000)

        rows.append({
            "step":        random.randint(1, 720),
            "type":        tx_type,
            "amount":      amount,
            "nameOrig":    sender,
            "nameDest":    receiver,
            "timestamp":   (base_time + timedelta(hours=random.randint(0, 720))).isoformat(),
            "isFraud":     is_fraud,
            "isFlaggedFraud": 1 if (is_fraud and amount > 200000) else 0,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(RAW_DIR, "paysim_transactions.csv")
    df.to_csv(path, index=False)
    print(f"[PaySim]    {len(df):,} rows → {path}")
    return df


# ── 3. Mock Govt Fraud Alerts (JSON) ─────────────────────────────────────────
def generate_govt_alerts(n=80):
    account_ids = [f"ACC_{i:05d}" for i in range(500)]
    alert_types = [
        "Mule Account Suspected",
        "Phishing-linked Account",
        "Cross-border Suspicious Transfer",
        "UPI Fraud Alert",
        "Cyber Crime Ticket",
    ]
    base_time = datetime(2024, 1, 1)

    alerts = []
    for i in range(n):
        alerts.append({
            "alert_id":       f"GOV_ALERT_{i:04d}",
            "account_id":     random.choice(account_ids),
            "alert_type":     random.choice(alert_types),
            "severity":       random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
            "issued_at":      (base_time + timedelta(days=random.randint(0, 180))).isoformat(),
            "description":    f"Regulatory alert issued for suspicious activity pattern #{random.randint(1000,9999)}",
            "resolved":       random.choice([True, False]),
        })

    path = os.path.join(RAW_DIR, "govt_fraud_alerts.json")
    with open(path, "w") as f:
        json.dump(alerts, f, indent=2)
    print(f"[Govt Alerts] {len(alerts)} alerts → {path}")
    return alerts


if __name__ == "__main__":
    print("Generating synthetic datasets...\n")
    generate_ieee_cis()
    generate_paysim()
    generate_govt_alerts()
    print("\nDone. All raw data saved to data/raw/")
