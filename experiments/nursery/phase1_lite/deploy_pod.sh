#!/bin/bash
# Deploy Phase 1-lite to a GPU pod and launch it.
#   usage: ./deploy_pod.sh root@<POD_IP> <SSH_PORT>
# Ships code + the 26MB phase-0 encoder checkpoint (not the 1.4GB rollout data).
set -e
HOST=$1; PORT=${2:-22}
NURSERY="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new $HOST"

echo "=== sync code + checkpoint ==="
$SSH "mkdir -p /workspace/nursery/phase0/ckpt /workspace/nursery/phase1_lite"
scp -P $PORT "$NURSERY"/phase0/{gallery_env.py,collect.py,train_wm.py,extract_reground.py} \
    "$HOST":/workspace/nursery/phase0/
scp -P $PORT "$NURSERY"/phase0/ckpt/wm_recon.pt "$HOST":/workspace/nursery/phase0/ckpt/
scp -P $PORT "$NURSERY"/phase1_lite/{env_constructs.py,prep_head.py,train_agent.py,eval_video.py,run_phase1_lite_pod.sh} \
    "$HOST":/workspace/nursery/phase1_lite/

echo "=== install deps + launch (nohup) ==="
$SSH 'cd /workspace/nursery/phase1_lite && \
  pip install -q open_clip_torch datasets scikit-learn matplotlib pillow && \
  chmod +x run_phase1_lite_pod.sh && \
  nohup bash run_phase1_lite_pod.sh > pipeline.log 2>&1 & \
  echo "launched: tail -f /workspace/nursery/phase1_lite/pipeline.log"'

echo
echo "check progress : ssh -p $PORT $HOST 'tail -5 /workspace/nursery/phase1_lite/run_grounded.log'"
echo "fetch results  : scp -P $PORT -r $HOST:/workspace/nursery/phase1_lite/results $NURSERY/phase1_lite/"
echo "                 scp -P $PORT -r $HOST:/workspace/nursery/phase1_lite/{logs,policies} $NURSERY/phase1_lite/"
