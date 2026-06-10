"""
test_fraudshield.py
FraudShield — Automated Test Suite

Run with: py -m pytest tests/ -v

Covers:
  - Ingestion pipeline schema and output
  - ML model scoring range and flag rate
  - Graph detector output structure
  - Config loading and path resolution
  - LLM explainer error handling
"""

import sys
import os
import json
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def sample_transactions():
    """Generate a small synthetic transaction DataFrame for testing."""
    from ingestion.generate_sample_data import generate_ieee_cis, generate_paysim
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        ieee_path = os.path.join(tmpdir, "ieee.csv")
        ps_path   = os.path.join(tmpdir, "paysim.csv")

        # Patch RAW_DIR temporarily
        import ingestion.generate_sample_data as gsd
        orig = gsd.RAW_DIR
        gsd.RAW_DIR = tmpdir
        generate_ieee_cis(n=200)
        generate_paysim(n=200)
        gsd.RAW_DIR = orig

        ieee_df = pd.read_csv(os.path.join(tmpdir, "ieee_cis_transactions.csv"))
        ps_df   = pd.read_csv(os.path.join(tmpdir, "paysim_transactions.csv"))
    return ieee_df, ps_df


@pytest.fixture(scope="session")
def sample_alerts():
    return pd.DataFrame([
        {"account_id": "ACC_00001", "severity": "HIGH",   "resolved": False,
         "issued_at": "2024-01-01", "alert_type": "Mule", "description": "Test"},
        {"account_id": "ACC_00002", "severity": "CRITICAL","resolved": False,
         "issued_at": "2024-01-02", "alert_type": "Fraud", "description": "Test"},
    ])


@pytest.fixture(scope="session")
def unified_df(sample_transactions, sample_alerts):
    """Build a unified transaction DataFrame through the pipeline."""
    import tempfile, os
    from ingestion.pipeline import load_ieee_cis, load_paysim, add_time_features
    from ingestion.pipeline import add_velocity_features, cross_reference_alerts

    ieee_raw, ps_raw = sample_transactions
    with tempfile.TemporaryDirectory() as tmpdir:
        ieee_path = os.path.join(tmpdir, "ieee.csv")
        ps_path   = os.path.join(tmpdir, "paysim.csv")
        ieee_raw.to_csv(ieee_path, index=False)
        ps_raw.to_csv(ps_path, index=False)

        ieee_df = load_ieee_cis(ieee_path)
        ps_df   = load_paysim(ps_path)

    df = pd.concat([ieee_df, ps_df], ignore_index=True)
    df = add_time_features(df)
    df = add_velocity_features(df)
    df = cross_reference_alerts(df, sample_alerts)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Config Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_config_imports(self):
        import config
        assert hasattr(config, "RISK_THRESHOLD")
        assert hasattr(config, "ROOT_DIR")
        assert hasattr(config, "PROCESSED_DIR")

    def test_thresholds_in_range(self):
        import config
        assert 0 < config.RISK_THRESHOLD < 1
        assert 0 < config.MULE_THRESHOLD_SUSPECT < 1
        assert config.MULE_THRESHOLD_SUSPECT < config.MULE_THRESHOLD_HIGH

    def test_directories_created(self):
        import config
        assert config.RAW_DIR.exists()
        assert config.PROCESSED_DIR.exists()
        assert config.MODEL_DIR.exists()

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key_123")
        import importlib, config
        importlib.reload(config)
        assert config.ANTHROPIC_API_KEY == "test_key_123"


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestion:
    def test_unified_schema_columns(self, unified_df):
        required = ["transaction_id", "timestamp", "sender_id", "receiver_id",
                    "amount", "tx_type", "source", "is_fraud"]
        for col in required:
            assert col in unified_df.columns, f"Missing column: {col}"

    def test_no_negative_amounts(self, unified_df):
        assert (unified_df["amount"] >= 0).all(), "Negative amounts found"

    def test_fraud_labels_binary(self, unified_df):
        assert set(unified_df["is_fraud"].unique()).issubset({0, 1})

    def test_time_features_present(self, unified_df):
        for col in ["hour_of_day", "day_of_week", "is_weekend", "is_night"]:
            assert col in unified_df.columns

    def test_is_weekend_binary(self, unified_df):
        assert set(unified_df["is_weekend"].unique()).issubset({0, 1})

    def test_is_night_binary(self, unified_df):
        assert set(unified_df["is_night"].unique()).issubset({0, 1})

    def test_govt_alert_column(self, unified_df):
        assert "govt_alert" in unified_df.columns
        assert set(unified_df["govt_alert"].unique()).issubset({0, 1})

    def test_alert_severity_values(self, unified_df):
        valid = {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert set(unified_df["alert_severity"].unique()).issubset(valid)

    def test_sources_correct(self, unified_df):
        assert set(unified_df["source"].unique()).issubset({"ieee_cis", "paysim"})

    def test_no_null_transaction_ids(self, unified_df):
        assert unified_df["transaction_id"].notna().all()

    def test_row_count(self, unified_df):
        assert len(unified_df) == 400  # 200 ieee + 200 paysim


# ══════════════════════════════════════════════════════════════════════════════
# ML Detection Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMLDetection:
    @pytest.fixture(scope="class")
    def scored_df(self, unified_df):
        from detection.model import preprocess, train_isolation_forest
        from detection.model import iso_score, composite_score, FEATURES
        from sklearn.model_selection import train_test_split
        import xgboost as xgb

        df_proc, _, _, _ = preprocess(unified_df)
        X = df_proc[FEATURES]
        y = df_proc["is_fraud"]

        iso = train_isolation_forest(X)
        iso_scores = iso_score(iso, X)

        # Simple XGB for test
        X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = xgb.XGBClassifier(n_estimators=10, verbosity=0, random_state=42)
        clf.fit(X_tr, y_tr)
        xgb_proba = clf.predict_proba(X)[:, 1]

        risk = composite_score(iso_scores, xgb_proba, df_proc["govt_alert"].values)
        result = unified_df.copy()
        result["iso_score"]  = iso_scores
        result["xgb_proba"]  = xgb_proba
        result["risk_score"] = risk
        result["ml_flagged"] = (risk >= 0.40).astype(int)
        return result

    def test_risk_score_range(self, scored_df):
        assert scored_df["risk_score"].between(0, 1).all(), "Risk scores out of [0,1]"

    def test_iso_score_range(self, scored_df):
        assert scored_df["iso_score"].between(0, 1).all()

    def test_xgb_proba_range(self, scored_df):
        assert scored_df["xgb_proba"].between(0, 1).all()

    def test_ml_flagged_binary(self, scored_df):
        assert set(scored_df["ml_flagged"].unique()).issubset({0, 1})

    def test_some_flagged(self, scored_df):
        assert scored_df["ml_flagged"].sum() > 0, "Nothing was flagged"

    def test_flag_rate_reasonable(self, scored_df):
        flag_rate = scored_df["ml_flagged"].mean()
        assert 0.0 < flag_rate < 0.80, f"Nothing was flagged or flag rate too high: {flag_rate:.2%}"


# ══════════════════════════════════════════════════════════════════════════════
# Graph Detection Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphDetection:
    @pytest.fixture(scope="class")
    def graph_results(self, unified_df):
        from graph.graph_detector import TransactionGraph, compute_fan_scores
        from graph.graph_detector import compute_mule_scores, compute_pass_through_ratio

        tg = TransactionGraph()
        tg.add_transactions(unified_df)
        G = tg.G

        bc = {n: 0.01 for n in G.nodes()}  # skip full betweenness for speed
        pt = compute_pass_through_ratio(unified_df)
        fn = compute_fan_scores(G)

        govt_flagged = set(unified_df[unified_df["govt_alert"] == 1]["sender_id"].tolist())
        mule_df = compute_mule_scores(G, unified_df, bc, pt, fn, govt_flagged)
        return mule_df, G

    def test_graph_has_nodes(self, graph_results):
        _, G = graph_results
        assert G.number_of_nodes() > 0

    def test_graph_has_edges(self, graph_results):
        _, G = graph_results
        assert G.number_of_edges() > 0

    def test_mule_score_range(self, graph_results):
        mule_df, _ = graph_results
        assert mule_df["mule_score"].between(0, 1).all()

    def test_mule_labels_valid(self, graph_results):
        mule_df, _ = graph_results
        valid = {"CLEAN", "SUSPECT", "HIGH_RISK"}
        assert set(mule_df["mule_label"].unique()).issubset(valid)

    def test_mule_df_columns(self, graph_results):
        mule_df, _ = graph_results
        for col in ["account_id", "mule_score", "mule_label",
                    "pass_through_ratio", "betweenness", "in_degree", "out_degree"]:
            assert col in mule_df.columns

    def test_pass_through_ratio_range(self, graph_results):
        mule_df, _ = graph_results
        assert mule_df["pass_through_ratio"].between(0, 1).all()


# ══════════════════════════════════════════════════════════════════════════════
# LLM Explainer Error Handling Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMExplainer:
    def test_missing_api_key_raises(self):
        """_resolve_api_key should raise ValueError when no key is available."""
        from llm.explainer import _resolve_api_key
        import pytest
        with pytest.raises(ValueError, match="Anthropic API key not found"):
            _resolve_api_key(api_key=None)

    def test_network_timeout_handled(self):
        """Explainer should handle timeout gracefully when key is provided."""
        import requests
        from unittest.mock import patch

        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            from llm.explainer import call_claude
            result = call_claude("test prompt", api_key="test_key")
            assert isinstance(result, dict)
            assert "explanation" in result
            assert "timed out" in result["explanation"]

    def test_malformed_json_response_handled(self):
        """Explainer should handle non-JSON responses without crashing."""
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "content": [{"text": "This is not JSON at all"}]
        }

        with patch("requests.post", return_value=mock_resp):
            from llm.explainer import call_claude
            result = call_claude("test prompt", api_key="test_key")
            assert isinstance(result, dict)
            assert "explanation" in result


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
