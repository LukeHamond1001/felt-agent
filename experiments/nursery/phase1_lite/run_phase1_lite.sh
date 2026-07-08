#!/bin/bash
# Phase 1-lite full matrix: prep R_phi, 3 conditions x 3 seeds, eval + demo GIF.
set -e
cd "$(dirname "$0")"
if [ ! -f ../phase0/ckpt/wm_recon.pt ]; then
  echo "ERROR: ../phase0/ckpt/wm_recon.pt not found — run ../phase0/run_phase0_local.sh first (Phase 0 outputs are git-ignored)."
  exit 1
fi
python3 prep_head.py
for cond in grounded rnd nohab; do
  for seed in 0 1 2; do
    echo "=== $cond seed $seed ==="
    python3 train_agent.py --reward $cond --seed $seed --steps 2000000
  done
done
python3 eval_video.py
echo "=== PHASE 1-LITE DONE ==="
