# !/usr/bin/env bash
set -uo pipefail

OUT_DIR="results/$(date +%Y%m%d-%H%M%S)-dqn-training"
mkdir -p "$OUT_DIR"

echo "=============================="
echo "DQN Training"
echo "$(date -u +%FT%T%z)"
echo "Output dir: $OUT_DIR"
echo "=============================="

python manifests/autoscaling/dqn/dqn_live_controller.py \
  --train-dir manifests/autoscaling/dqn/training_data \
  --log "$OUT_DIR/dqn_live_training_curve.csv" \
  --save-model manifests/autoscaling/dqn/dqn_model.pth \
  --train-only

echo "=============================="
echo "Training Completed"
echo "Results saved to: $OUT_DIR"
echo "=============================="
