#!/usr/bin/env bash
set -uo pipefail

echo "========================================="
echo "  ARIMA Live Scaling Experiment"
echo "  $(date -u +%FT%T%z)"
echo "========================================="

NS="open5gs"
UE_POD=$(kubectl get pod -n "$NS" -l name=ue1 \
         -o jsonpath='{.items[0].metadata.name}')

# Ensure iperf3 server is running
pkill iperf3 2>/dev/null; sleep 1
iperf3 -s -D -p 5201
echo "[setup] iperf3 server started"

# Start ARIMA live controller (NOT dry-run — will actually scale!)
echo "[setup] Starting ARIMA live controller..."
python manifests/autoscaling/arima/arima_live_controller.py \
  --interval 5 \
  --threshold 4000 \
  --horizon 3 \
  --cooldown 20 \
  --log manifests/autoscaling/arima/results/arima_live_experiment.csv &
CTRL_PID=$!
echo "[setup] Controller PID: $CTRL_PID"

# Wait for controller to initialize and collect baseline
sleep 20

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
    sleep 10
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

# Stop controller
kill $CTRL_PID 2>/dev/null
wait $CTRL_PID 2>/dev/null

echo ""
echo "========================================="
echo "  ARIMA Live Experiment — Complete"
echo "  $(date -u +%FT%T%z)"
echo "========================================="
echo ""
echo "=== Results CSV ==="
cat manifests/autoscaling/arima/results/arima_live_experiment.csv
