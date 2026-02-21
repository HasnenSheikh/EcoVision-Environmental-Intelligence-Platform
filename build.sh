#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  EcoVision – Render Build Script
#  Runs once per deploy.  install deps → ensure dirs → train if needed
# ─────────────────────────────────────────────────────────────
set -o errexit   # exit immediately on any error

echo "========================================================"
echo "  EcoVision  |  Render Build  |  $(date)"
echo "========================================================"

# ── 1. Upgrade pip silently ──────────────────────────────────
echo "[1/3] Upgrading pip..."
pip install --upgrade pip --quiet

# ── 2. Install all Python dependencies ──────────────────────
echo "[2/3] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# ── 3. Ensure runtime directories exist ─────────────────────
echo "[3/3] Preparing runtime directories..."
mkdir -p data models

# ── 4. Train models only when pkl files are missing ─────────
MODELS_NEEDED=false
for f in "models/energy_lstm.pkl" "models/waste_rf.pkl" "models/water_lstm.pkl"; do
  if [ ! -f "$f" ]; then
    MODELS_NEEDED=true
    echo "  Missing: $f"
  fi
done

if [ "$MODELS_NEEDED" = true ]; then
  echo ""
  echo ">>> Pre-trained models not found – running train_models.py"
  echo "    (This takes ~3-5 minutes on first deploy)"
  python train_models.py
else
  echo "  Pre-trained models found – skipping training."
fi

echo ""
echo "========================================================"
echo "  Build complete."
echo "========================================================"
