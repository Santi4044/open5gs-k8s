# !/usr/bin/env bash
set -uo pipefail

OUT_DIR="${OUT_DIR:-results/$(date +%Y%m%d-%H%M%S)-dqn-experiment}"
mkdir -p "$OUT_DIR"

echo "================================="
echo "DQN Live Scaling Experiment"
echo "$(date -u +%FT%T%z)"
echo "Output dir: $OUT_DIR"
echo "================================="

# Ensure iperf3 server is running
pkill iperf3 2>/dev/null; sleep 1
iperf3 -s -D -p 5201
echo "iperf3 server started"

# Start the DQN live controller with a pre-trained model
echo "Starting DQN live controller (pre-trained model)..."
python manifests/autoscaling/dqn/dqn_live_controller.py \
  --interval 5 \
  --threshold 1500 \
  --cooldown 30 \
  --load-model manifests/autoscaling/dqn/dqn_model.pth \
  --log "$OUT_DIR/dqn_live.csv" &
CTRL_PID=$!
echo "Controller PID: $CTRL_PID"

# Wait for the controller to initialise
echo "Waiting for controller to initialise..."
sleep 10

# Run traffic
OUT_LOG="$OUT_DIR/traffic.log" scripts/traffic/run_traffic_phases.sh

# Let controller observe cooldown
echo ""
echo "All phases complete. Waiting 30s for controller to stabilise..."
sleep 30

# Stop the controller
kill $CTRL_PID 2>/dev/null
wait $CTRL_PID 2>/dev/null
echo ""
echo "=================================="
echo "DQN Live Experiment - Completed"
echo "$(date -u +%FT%T%z)"
echo "=================================="
echo ""
echo "Results saved to: $OUT_DIR"
echo "=== Results CSV ==="
cat "$OUT_DIR/dqn_live.csv"
