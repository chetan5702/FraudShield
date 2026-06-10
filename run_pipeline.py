"""
run_pipeline.py
FraudShield — Full Pipeline Bootstrap

Run this once to generate data and train models before launching the dashboard.
Usage: python run_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "═"*55)
print("  FraudShield — Pipeline Bootstrap")
print("═"*55 + "\n")

# Step 1: Generate synthetic data
print("STEP 1/3 — Generating synthetic datasets...")
from ingestion.generate_sample_data import generate_ieee_cis, generate_paysim, generate_govt_alerts
generate_ieee_cis()
generate_paysim()
generate_govt_alerts()

# Step 2: Run ingestion pipeline
print("\nSTEP 2/3 — Running ingestion & normalization pipeline...")
from ingestion.pipeline import run_pipeline
df = run_pipeline()

# Step 3: Run ML detection
print("\nSTEP 3/3 — Training ML models & scoring transactions...")
from detection.model import run_detection
scored_df = run_detection(df)

# Step 4: Run graph detection
print("\nSTEP 4/4 — Running graph-based mule detection...")
from graph.graph_detector import run_graph_detection
mule_df, G = run_graph_detection(scored_df)

print("\n" + "═"*55)
print("  ✅ Bootstrap complete!")
print("  Launch dashboard with:")
print("     streamlit run dashboard/app.py")
print("═"*55 + "\n")
