#!/bin/bash
# Pod variant: the GPU is barely loaded (env stepping is CPU), so run the three
# conditions as parallel processes, seeds sequential within each. ~1-2 h on a 4090.
set -e
cd "$(dirname "$0")"
python3 prep_head.py
for cond in grounded rnd nohab; do
  (
    for seed in 0 1 2; do
      echo "=== $cond seed $seed ==="
      python3 train_agent.py --reward $cond --seed $seed --steps 2000000
    done
  ) > run_$cond.log 2>&1 &
done
wait
python3 eval_video.py
echo "=== PHASE 1-LITE DONE ==="
