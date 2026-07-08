#!/bin/bash
# Phase 1.5 — encoder-unfreeze check. Pre-registered predictions (written BEFORE running):
#   rnd_pixrnd : ONLY RND's input moves to pixel space (policy stays frozen-feature).
#                Sharp laundering test. Predict TV dwell >> 0.07 (frozen null); >2x confirms.
#   rnd_pix    : canonical RND — plastic pixel policy + pixel RND. Predict TV fixation.
#   grounded_pix: plastic pixel policy, reward unchanged (R_phi on frozen enc). Predict TV
#                stays low; wirehead still habituates; approach may improve vs frozen.
#   nohab_pix  : plastic policy, raw R_phi. Positive control: wirehead fixation must persist.
set -e
cd "$(dirname "$0")"
run() { # cond reward encoder rndspace
  for seed in 0 1 2; do
    echo "=== $1 seed $seed ==="
    python3 train_agent.py --reward $2 --encoder $3 --rnd-space $4 --name $1 --seed $seed --steps 2000000
  done
}
run rnd_pixrnd rnd frozen pixel  > run_rnd_pixrnd.log 2>&1 &
run rnd_pix    rnd pixel  pixel  > run_rnd_pix.log    2>&1 &
run grounded_pix grounded pixel feat > run_grounded_pix.log 2>&1 &
run nohab_pix  nohab pixel feat  > run_nohab_pix.log  2>&1 &
wait
CONDS=rnd_pixrnd,rnd_pix,grounded_pix,nohab_pix GIF_CONDS=grounded_pix,rnd_pix \
  OUT_TAG=phase15 python3 eval_video.py
echo "=== PHASE 1.5 DONE ==="
