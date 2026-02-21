"""
EcoVision – Baseline Model Training Script
==========================================
Run once to train and save all baseline models.

Usage:
    python train_models.py

Outputs (models/ directory):
    energy_prophet.pkl   – Prophet daily energy forecaster
    energy_lstm.pkl      – LSTM fallback  (only if Prophet MAPE > 10 %)
    waste_rf.pkl         – Random Forest waste forecaster
    training_metrics.json

The Flask app auto-loads these at startup via ModelRegistry.
"""

import sys
import os
import time

# Make sure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_pipeline  import DataPipeline
from utils.model_trainer  import train_all_models


def main():
    t0 = time.time()
    print("=" * 60)
    print("  EcoVision  │  Baseline Model Training")
    print(f"  Date : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Step 1: Run data pipeline ─────────────────────────
    print("\n[1/2]  Running data pipeline …")
    pipeline = DataPipeline(dataset_path='dataset')
    ok = pipeline.run()
    if not ok:
        print(f"\n  ❌ Pipeline errors: {pipeline.load_errors}")
        sys.exit(1)

    health = pipeline.health_report()
    print("\n  Dataset health:")
    for name, stats in health.items():
        status = "✅" if stats.get('status') == 'ok' else "⚠"
        print(
            f"    {status} {name:20s}  "
            f"rows={stats.get('rows', '?'):>5}  "
            f"nulls={stats.get('nulls', '?')}  "
            f"range={stats.get('date_range', '?')}"
        )

    # ── Step 2: Train models ───────────────────────────────
    print("\n[2/2]  Training models …")
    results = train_all_models(pipeline)

    # ── Done ──────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  ⏱  Total training time: {elapsed:.1f} s")
    print("\n  Models saved to  models/ ")
    print("  Restart Flask to pick up the new models.")
    print("=" * 60)


if __name__ == '__main__':
    main()
