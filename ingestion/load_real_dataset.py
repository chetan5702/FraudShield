"""
load_real_dataset.py
FraudShield — Hackathon Dataset Loader

Loads the anonymized 3924-feature dataset provided by the hackathon organizers.
Extracts readable metadata columns, selects numeric features for ML,
and outputs a unified processed CSV compatible with the detection pipeline.

Run: python ingestion/load_real_dataset.py --input DataSet.csv
"""

import os
import argparse
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Known metadata columns (by position discovered via inspection)
META_COLS = {
    "F2230": "period",           # 'Oct25' — reporting period
    "F3886": "account_type",     # 'Savings'
    "F3889": "product_code",     # 'G365D'
    "F3890": "risk_rating",      # 'R', 'SU'
    "F3891": "occupation",       # 'selfemployed', 'student'
    "F3892": "gender",           # 'M'
    "F3893": "channel",          # 'RETAIL'
}


def load_dataset(path: str) -> pd.DataFrame:
    print(f"[Load]  Reading dataset from {path}...")
    df = pd.read_csv(path, index_col=0, low_memory=False)
    print(f"[Load]  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return df


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Pull out human-readable metadata columns."""
    meta = pd.DataFrame(index=df.index)
    for col, name in META_COLS.items():
        if col in df.columns:
            meta[name] = df[col]
    meta["account_id"] = [f"ACC_{str(i).zfill(6)}" for i in df.index]
    return meta


def select_numeric_features(df: pd.DataFrame, max_na_frac: float = 0.70) -> pd.DataFrame:
    """
    Select numeric columns only, drop columns with >70% NAs,
    fill remaining NAs with column median, clip extreme outliers.
    """
    # Drop known metadata/string columns
    drop_cols = list(META_COLS.keys())
    df_num = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert to numeric (coerce non-numeric to NaN)
    df_num = df_num.apply(pd.to_numeric, errors="coerce")

    # Drop high-NA columns
    na_frac = df_num.isna().mean()
    keep = na_frac[na_frac <= max_na_frac].index
    df_num = df_num[keep]
    print(f"[Features] Kept {len(keep):,} / {df.shape[1]} columns (≤{int(max_na_frac*100)}% NA)")

    # Fill NAs with median
    df_num = df_num.fillna(df_num.median())

    # Clip extreme outliers (±5 std dev per column)
    for col in df_num.columns:
        mu, sigma = df_num[col].mean(), df_num[col].std()
        if sigma > 0:
            df_num[col] = df_num[col].clip(mu - 5*sigma, mu + 5*sigma)

    return df_num


def engineer_risk_proxy(df_num: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """
    Create a rough fraud proxy label using domain heuristics
    (since dataset has no explicit fraud label):
    - High-variance accounts (many large swings in numeric features)
    - Accounts with 'SU' (Suspicious) risk rating
    """
    # Variance-based proxy: top 5% most volatile rows
    row_std = df_num.std(axis=1)
    high_var = (row_std > row_std.quantile(0.95)).astype(int)

    # Risk rating proxy
    sus_rating = (meta.get("risk_rating", pd.Series("R", index=meta.index)) == "SU").astype(int)

    proxy = ((high_var + sus_rating) >= 1).astype(int)
    print(f"[Proxy] Suspicious proxy labels: {proxy.sum():,} / {len(proxy):,} ({proxy.mean()*100:.1f}%)")
    return proxy


def run(input_path: str, save: bool = True) -> pd.DataFrame:
    print("=" * 55)
    print("  FraudShield — Hackathon Dataset Ingestion")
    print("=" * 55)

    df_raw  = load_dataset(input_path)
    meta    = extract_metadata(df_raw)
    df_num  = select_numeric_features(df_raw)
    proxy   = engineer_risk_proxy(df_num, meta)

    # Build output dataframe
    out = meta.copy()
    out["is_fraud"]    = proxy
    out["row_std"]     = df_raw.apply(pd.to_numeric, errors="coerce").std(axis=1).values
    out["na_count"]    = df_raw.isnull().sum(axis=1).values
    out["na_fraction"] = out["na_count"] / df_raw.shape[1]

    # Attach reduced numeric features (top 50 by variance for ML)
    top_features = df_num.var().nlargest(50).index.tolist()
    feature_df   = df_num[top_features].copy()
    feature_df.columns = [f"feat_{i}" for i in range(len(top_features))]
    out = pd.concat([out.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    # Summary
    print(f"\n[Summary]")
    print(f"  Total accounts       : {len(out):,}")
    print(f"  Suspicious (proxy)   : {out['is_fraud'].sum():,}")
    print(f"  Account types        : {out['account_type'].value_counts().to_dict() if 'account_type' in out else 'N/A'}")
    print(f"  Risk ratings         : {out['risk_rating'].value_counts().to_dict() if 'risk_rating' in out else 'N/A'}")
    print(f"  Occupations          : {out['occupation'].value_counts().to_dict() if 'occupation' in out else 'N/A'}")
    print(f"  Numeric features kept: {len(top_features)} (top 50 by variance)")

    if save:
        out_path = os.path.join(PROCESSED_DIR, "real_dataset_processed.csv")
        out.to_csv(out_path, index=False)

        # Also save full numeric matrix for model training
        full_path = os.path.join(PROCESSED_DIR, "real_dataset_features.csv")
        df_num.reset_index(drop=True).to_csv(full_path, index=False)

        print(f"\n[Saved]  Processed → {out_path}")
        print(f"[Saved]  Full features → {full_path}")

    print("=" * 55)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to DataSet.csv")
    args = parser.parse_args()
    result = run(args.input)
    print(f"\nSample output:")
    print(result[["account_id", "account_type", "risk_rating", "occupation",
                   "is_fraud", "na_fraction", "row_std"]].head(10).to_string(index=False))
