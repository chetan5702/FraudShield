"""
explainer.py
FraudShield — LLM Explainability Layer

Supports two LLM providers (auto-detected from environment):
  1. Google Gemini  — free tier, default (GEMINI_API_KEY)
  2. Anthropic Claude — pay-per-use fallback (ANTHROPIC_API_KEY)

Priority: Gemini → Claude → error

Two explanation types:
  1. Transaction explanation  — why this transaction was flagged
  2. Account explanation      — why this account is suspected as a mule
"""

import os
import json
import requests
import pandas as pd

PROCESSED_DIR    = os.path.join(os.path.dirname(__file__), "../data/processed")
ANTHROPIC_URL    = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL  = "claude-sonnet-4-20250514"
GEMINI_MODEL     = "gemini-1.5-flash"
GEMINI_URL       = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_PROMPT = """You are a financial fraud analyst AI assistant for FraudShield,
an AI-powered fraud detection system used by banks and regulators.

Your job is to generate concise, professional, auditor-ready risk narratives
explaining why a transaction or account has been flagged as suspicious.

Rules:
- Be specific — cite the actual numbers and patterns from the data
- Be concise — max 3 sentences per explanation
- Use professional banking/compliance language
- Always end with a clear risk verdict: LOW / MEDIUM / HIGH / CRITICAL
- Return ONLY valid JSON with keys: "explanation" and "verdict"
- No markdown, no backticks, no preamble — just the raw JSON object
"""


def _resolve_api_key(api_key: str = None) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Anthropic API key not found. Either pass api_key= explicitly or "
            "set the ANTHROPIC_API_KEY environment variable."
        )
    return key


def _resolve_gemini_key(api_key: str = None) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
    return key


def _parse_response(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {"explanation": raw.strip(), "verdict": "UNKNOWN"}


# ── Gemini Call ───────────────────────────────────────────────────────────────

def call_gemini(prompt: str, api_key: str = None) -> dict:
    """Call Gemini API and return parsed JSON response."""
    key = _resolve_gemini_key(api_key)
    try:
        response = requests.post(
            f"{GEMINI_URL}?key={key}",
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 500,
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        raw  = data["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_response(raw)
    except requests.exceptions.Timeout:
        return {"explanation": "Explanation unavailable: request timed out.", "verdict": "UNKNOWN"}
    except requests.exceptions.HTTPError as e:
        return {"explanation": f"Explanation unavailable: API error {e.response.status_code}.", "verdict": "UNKNOWN"}
    except requests.exceptions.RequestException as e:
        return {"explanation": f"Explanation unavailable: {e}", "verdict": "UNKNOWN"}
    except (KeyError, IndexError):
        return {"explanation": "Explanation unavailable: unexpected API response format.", "verdict": "UNKNOWN"}


# ── Claude Call ───────────────────────────────────────────────────────────────

def call_claude(prompt: str, api_key: str = None) -> dict:
    """Call Claude API and return parsed JSON response."""
    key = _resolve_api_key(api_key)
    try:
        response = requests.post(
            ANTHROPIC_URL,
            headers={"Content-Type": "application/json", "x-api-key": key},
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1000,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if "content" not in data or not data["content"]:
            return {"explanation": "Explanation unavailable: unexpected API response format.", "verdict": "UNKNOWN"}
        return _parse_response(data["content"][0]["text"])
    except requests.exceptions.Timeout:
        return {"explanation": "Explanation unavailable: request timed out.", "verdict": "UNKNOWN"}
    except requests.exceptions.HTTPError as e:
        return {"explanation": f"Explanation unavailable: API error {e.response.status_code}.", "verdict": "UNKNOWN"}
    except requests.exceptions.RequestException as e:
        return {"explanation": f"Explanation unavailable: {e}", "verdict": "UNKNOWN"}


# ── Auto-routing ──────────────────────────────────────────────────────────────

def call_llm(prompt: str, gemini_key: str = None, claude_key: str = None) -> dict:
    """
    Auto-route to available LLM provider.
    Priority: Gemini (free) → Claude (paid)
    """
    # Try Gemini first
    g_key = gemini_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if g_key:
        return call_gemini(prompt, api_key=g_key)

    # Fallback to Claude
    c_key = claude_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if c_key:
        return call_claude(prompt, api_key=c_key)

    return {
        "explanation": "No LLM API key configured. Set GEMINI_API_KEY or ANTHROPIC_API_KEY.",
        "verdict": "UNKNOWN"
    }



# ── Transaction Explainer ─────────────────────────────────────────────────────

def explain_transaction(row: pd.Series, mule_scores: dict = None, api_key: str = None) -> dict:
    sender_mule = mule_scores.get(row["sender_id"], {}) if mule_scores else {}
    receiver_mule = mule_scores.get(row["receiver_id"], {}) if mule_scores else {}

    prompt = f"""
Explain why the following transaction was flagged as suspicious:

Transaction Details:
- Transaction ID   : {row['transaction_id']}
- Timestamp        : {row['timestamp']}
- Sender Account   : {row['sender_id']}
- Receiver Account : {row['receiver_id']}
- Amount           : ₹{row['amount']:,.2f}
- Transaction Type : {row['tx_type']}
- Hour of Day      : {row.get('hour_of_day', 'N/A')} (is_night: {row.get('is_night', 'N/A')})
- Sender tx count (24h): {row.get('tx_count_24h', 'N/A')}
- Amount Z-score   : {row.get('amount_zscore', 'N/A'):.2f} (deviation from sender's avg)
- Govt Alert Flag  : {'Yes' if row.get('govt_alert') else 'No'} (severity: {row.get('alert_severity', 'NONE')})
- ML Risk Score    : {row.get('risk_score', 'N/A'):.3f}
- XGBoost Fraud Prob: {row.get('xgb_proba', 'N/A'):.3f}
- Isolation Forest Score: {row.get('iso_score', 'N/A'):.3f}

Sender Mule Profile:
- Mule Score       : {sender_mule.get('mule_score', 'N/A')}
- Mule Label       : {sender_mule.get('mule_label', 'N/A')}
- Pass-through Ratio: {sender_mule.get('pass_through_ratio', 'N/A')}

Receiver Mule Profile:
- Mule Score       : {receiver_mule.get('mule_score', 'N/A')}
- Mule Label       : {receiver_mule.get('mule_label', 'N/A')}

Return JSON with keys "explanation" and "verdict".
"""
    return call_claude(prompt, api_key)


# ── Account Explainer ─────────────────────────────────────────────────────────

def explain_account(account_row: pd.Series, tx_df: pd.DataFrame, api_key: str = None) -> dict:
    acc = account_row["account_id"]
    acc_txns = tx_df[(tx_df["sender_id"] == acc) | (tx_df["receiver_id"] == acc)]

    total_sent     = acc_txns[acc_txns["sender_id"] == acc]["amount"].sum()
    total_received = acc_txns[acc_txns["receiver_id"] == acc]["amount"].sum()
    unique_senders = acc_txns[acc_txns["receiver_id"] == acc]["sender_id"].nunique()
    unique_receivers = acc_txns[acc_txns["sender_id"] == acc]["receiver_id"].nunique()

    prompt = f"""
Explain why the following bank account has been flagged as a suspected mule account:

Account Profile:
- Account ID         : {acc}
- Mule Score         : {account_row['mule_score']} / 1.0
- Mule Label         : {account_row['mule_label']}
- Betweenness Centrality: {account_row['betweenness']} (network bridge score)
- Pass-through Ratio : {account_row['pass_through_ratio']} (funds forwarded within 2hrs)
- Fan Score          : {account_row['fan_score']} (in/out degree asymmetry)
- Govt Alert Flagged : {'Yes' if account_row['govt_flagged'] else 'No'}

Transaction Summary:
- Total Received     : ₹{total_received:,.2f} from {unique_senders} unique senders
- Total Sent         : ₹{total_sent:,.2f} to {unique_receivers} unique receivers
- In-degree          : {account_row['in_degree']} (accounts sending to this account)
- Out-degree         : {account_row['out_degree']} (accounts receiving from this account)
- Total transactions : {len(acc_txns)}

Return JSON with keys "explanation" and "verdict".
"""
    return call_claude(prompt, api_key)

def run_explainer(
    n_transactions: int = 5,
    n_accounts: int = 3,
    save: bool = True,
    api_key: str = None,
) -> tuple:

    # Resolve key early so we fail fast before doing any compute
    key = _resolve_api_key(api_key)

    print("=" * 55)
    print("  FraudShield — LLM Explainability Layer")
    print("=" * 55)

    # Load scored data
    tx_df    = pd.read_csv(os.path.join(PROCESSED_DIR, "scored_transactions.csv"), parse_dates=["timestamp"])
    mule_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "mule_scores.csv"))

    # Build mule lookup
    mule_scores = mule_df.set_index("account_id").to_dict(orient="index")

    # Top flagged transactions
    flagged_txns = tx_df[tx_df["ml_flagged"] == 1].sort_values("risk_score", ascending=False).head(n_transactions)

    print(f"\n[Transactions] Explaining top {n_transactions} flagged transactions...")
    txn_explanations = []
    for _, row in flagged_txns.iterrows():
        result = explain_transaction(row, mule_scores, api_key=key)
        txn_explanations.append({
            "transaction_id": row["transaction_id"],
            "sender_id":      row["sender_id"],
            "receiver_id":    row["receiver_id"],
            "amount":         row["amount"],
            "risk_score":     row["risk_score"],
            "explanation":    result.get("explanation", ""),
            "verdict":        result.get("verdict", "UNKNOWN"),
        })
        print(f"  ✓ {row['transaction_id']}  →  Verdict: {result.get('verdict', '?')}")

    # Top mule accounts
    suspect_accounts = mule_df[mule_df["mule_label"].isin(["SUSPECT", "HIGH_RISK"])].head(n_accounts)

    print(f"\n[Accounts] Explaining top {n_accounts} suspect accounts...")
    acc_explanations = []
    for _, row in suspect_accounts.iterrows():
        result = explain_account(row, tx_df, api_key=key)
        acc_explanations.append({
            "account_id":  row["account_id"],
            "mule_score":  row["mule_score"],
            "mule_label":  row["mule_label"],
            "explanation": result.get("explanation", ""),
            "verdict":     result.get("verdict", "UNKNOWN"),
        })
        print(f"  ✓ {row['account_id']}  →  Verdict: {result.get('verdict', '?')}")

    txn_exp_df = pd.DataFrame(txn_explanations)
    acc_exp_df = pd.DataFrame(acc_explanations)

    if save:
        txn_exp_df.to_csv(os.path.join(PROCESSED_DIR, "transaction_explanations.csv"), index=False)
        acc_exp_df.to_csv(os.path.join(PROCESSED_DIR, "account_explanations.csv"), index=False)
        print(f"\n[Saved]  Explanations → data/processed/")

    print("=" * 55)
    return txn_exp_df, acc_exp_df


if __name__ == "__main__":
    import sys
    _key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not _key:
        print("Error: set ANTHROPIC_API_KEY environment variable before running.")
        sys.exit(1)
    txn_exp, acc_exp = run_explainer(n_transactions=5, n_accounts=3, api_key=_key)

    print("\n── Sample Transaction Explanation ──")
    if len(txn_exp):
        r = txn_exp.iloc[0]
        print(f"Transaction : {r['transaction_id']}")
        print(f"Amount      : ₹{r['amount']:,.2f}")
        print(f"Risk Score  : {r['risk_score']:.3f}")
        print(f"Explanation : {r['explanation']}")
        print(f"Verdict     : {r['verdict']}")

    print("\n── Sample Account Explanation ──")
    if len(acc_exp):
        r = acc_exp.iloc[0]
        print(f"Account     : {r['account_id']}")
        print(f"Mule Score  : {r['mule_score']}")
        print(f"Explanation : {r['explanation']}")
        print(f"Verdict     : {r['verdict']}")
