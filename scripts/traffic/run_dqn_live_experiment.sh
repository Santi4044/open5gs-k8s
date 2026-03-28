#!/usr/bin/env bash
set -uo pipefail

echo "========================================="
echo "  DQN Live Scaling Experiment"
echo "  $(date -u +%FT%T%z)"
echo "========================================="

NS="open5gs"
UE_POD=$(kubectl get pod -n "$NS" -l name=ue1 \
         -o jsonpath='{.items[0].metadata.name}')

# Ensure iperf3 server is running
pkill iperf3 2>/dev/null; sleep 1
iperf3 -s -D -p 5201
echo "[setup] iperf3 server started"

# Start DQN live controller with PRE-TRAINED model
echo "[setup] Starting DQN live controller (pre-trained model)..."
python manifests/autoscaling/dqn/dqn_live_controller.py \
  --interval 5 \
  --threshold 4000 \
  --cooldown 20 \
  --load-model manifests/autoscaling/dqn/dqn_model.pth \
  --log manifests/autoscaling/dqn/results/dqn_live_experiment.csv &
CTRL_PID=$!
echo "[setup] Controller PID: $CTRL_PID"

# Wait for controller to initialize
echo "[setup] Waiting for controller to initialize (~10s)..."
sleep 10

# Verify controller is still running
if ! kill -0 $CTRL_PID 2>/dev/null; then
  echo "[error] Controller died! Check logs."
  exit 1
fi
echo "[setup] Controller is live. Starting traffic phases..."

# Start persistent iperf3 server (restarts after each client)
start_iperf_server() {
  pkill -f "iperf3 -s" 2>/dev/null; sleep 1
  while true; do
    iperf3 -s -p 5201 --one-off 2>/dev/null
  done &
  IPERF_PID=$!
  echo "[setup] iperf3 persistent server started (PID: $IPERF_PID)"
}
start_iperf_server

# Run burst traffic pattern
run_phase() {
  local label="$1" bitrate="$2" dur="$3"
  echo ""
  echo "[traffic] $(date -u +%FT%T%z) === Phase: $label | bitrate=$bitrate | dur=${dur}s ==="
  if [ "$bitrate" = "0" ]; then
    sleep "$dur"
  else
    kubectl exec -n "$NS" "$UE_POD" -c ue -- \
      iperf3 -c 10.10.6.100 -p 5201 -u -b "$bitrate" -l 1200 -t "$dur" 2>&1 | tail -3
    sleep 15
  fi
}

run_phase "1-LOW"    20M  30
run_phase "2-HIGH"   50M  30
run_phase "3-LOW"    20M  30
run_phase "4-SPIKE"  60M  30
run_phase "5-IDLE"   0    30

# Let controller observe cooldown
echo ""
echo "[traffic] All phases complete. Waiting 30s for controller to stabilize..."
sleep 30

# Stop iperf3 server
kill $IPERF_PID 2>/dev/null

# Stop controller
kill $CTRL_PID 2>/dev/null
wait $CTRL_PID 2>/dev/null

echo ""
echo "========================================="
echo "  DQN Live Experiment — Complete"
echo "  $(date -u +%FT%T%z)"
echo "========================================="
echo ""
echo "=== Results CSV ==="
cat manifests/autoscaling/dqn/results/dqn_live_experiment.csv
