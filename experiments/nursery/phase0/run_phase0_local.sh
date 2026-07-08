#!/bin/zsh
# Phase 0 local pipeline — collection -> two world models -> re-grounding measurement.
set -e
cd "$(dirname "$0")"
echo "=== collect train (100k frames) ==="
python3 collect.py --mode train --frames 100000
echo "=== collect eval (25k frames, heldout-strong billboards only) ==="
python3 collect.py --mode eval --frames 25000
echo "=== train WM recon (12k steps) ==="
python3 train_wm.py --objective recon --steps 12000
echo "=== train WM jepa (12k steps) ==="
python3 train_wm.py --objective jepa --steps 12000
echo "=== re-ground + measure ==="
python3 extract_reground.py
echo "=== PHASE 0 PIPELINE DONE ==="
