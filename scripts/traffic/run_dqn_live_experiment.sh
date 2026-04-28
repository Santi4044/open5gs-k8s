#!/usr/bin/env bash
set -uo pipefail

OUT_DIR="${OUT_DIR:-results/$(date +%Y%m%d-%H%M%S)-dqn-experiment}"
mkdir -p "$OUT_DIR"

echo "========================================="
echo "  DQN Live Scaling Experiment"
echo "  $(date -u +%FT%T%z)"
echo "  Output dir: $OUT_DIR"
echo "========================================="

NS="open5gs"
UE_POD=$(kubectl get pod -n "$NS" -l name=ue1 \
         -o jsonpath='{.items[0].metadata.name}')

# Ensure iperf3 server is running
pkill iperf3 2>/dev/null; sleep 1
iperf3 -s -D -p 5201
echo "iperf3 server started"

# Start DQN live controller with PRE-TRAINED model
echo "Starting DQN live controller (pre-trained model)..."
python manifests/autoscaling/dqn/dqn_live_controller.py \
  --interval 5 \
  --threshold 1500 \
  --cooldown 30 \
  --load-model manifests/autoscaling/dqn/dqn_model.pth \
  --log "$OUT_DIR/dqn_live.csv" &
CTRL_PID=$!
echo "Controller PID: $CTRL_PID"

# Wait for controller to initialize
echo "Waiting for controller to initialize (~10s)..."
sleep 10

# Run burst traffic pattern
run_phase() {
  local label="$1" bitrate="$2" dur="$3"
  echo ""
  echo "$(date -u +%FT%T%z) === Phase: $label | bitrate=$bitrate | dur=${dur}s ==="
  if [ "$bitrate" = "0" ]; then
    sleep "$dur"
  else
    kubectl exec -n "$NS" "$UE_POD" -c ue -- \
      iperf3 -c 10.10.6.100 -p 5201 -u -b "$bitrate" -l 1200 -t "$dur" 2>&1 | tail -3
    sleep 15
  fi
}

run_phase "1-IDLE"   0    30
run_phase "2-LOW"    10M  60
run_phase "3-IDLE"   0    30
run_phase "4-HIGH"   40M  120
run_phase "5-IDLE"   0    120

# Let controller observe cooldown
echo ""
echo "All phases complete. Waiting 30s for controller to stabilize..."
sleep 30

# Stop controller
kill $CTRL_PID 2>/dev/null
wait $CTRL_PID 2>/dev/null
echo ""
echo "========================================="
echo "  DQN Live Experiment — Complete"
echo "  $(date -u +%FT%T%z)"
echo "========================================="
echo ""
echo "Results saved to: $OUT_DIR"
echo "=== Results CSV ==="
cat "$OUT_DIR/dqn_live.csv"