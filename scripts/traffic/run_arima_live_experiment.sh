#!/usr/bin/env bash
set -uo pipefail

#Output dir (timestamped)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="results/${TIMESTAMP}-arima-experiment"
mkdir -p "$OUT_DIR"
echo "[exp] Output dir: $OUT_DIR"

NS="open5gs"
UE_POD=$(kubectl get pod -n "$NS" -l name=ue1 \
         -o jsonpath='{.items[0].metadata.name}')

LOG_CSV="${OUT_DIR}/arima_live.csv"
TRAFFIC_LOG="${OUT_DIR}/traffic.log"

#Start ARIMA live controller
echo "[exp] Starting ARIMA live controller..."
python manifests/autoscaling/arima/arima_live_controller.py \
  --interval 5 \
  --threshold 1500 \
  --horizon 3 \
  --cooldown 30 \
  --window 30 \
  --min-window 10 \
  --log "$LOG_CSV" &
CTRL_PID=$!
echo "Controller PID: $CTRL_PID"

#Wait for controller to initialise
echo "Waiting 3s for controller to initialise..."

sleep 3

#Traffic phases
run_phase() {
  local label="$1" bitrate="$2" dur="$3"
  echo "Phase: $label | bitrate=$bitrate | dur=${dur}s" | tee -a "$TRAFFIC_LOG"
  sleep 1  # sync delay so phase label prints before traffic starts
  if [ "$bitrate" = "0" ]; then
    sleep "$dur"
  else
    kubectl exec -n "$NS" "$UE_POD" -c ue -- sh -lc "
      ip route add 10.10.6.0/24 dev uesimtun0 2>/dev/null || true
      iperf3 -u -c 10.10.6.100 -p 5201 -b ${bitrate} -l 1200 -t ${dur} --connect-timeout 5000 > /dev/null 2>&1
    " > /dev/null 2>&1
  fi
}

run_phase "1-IDLE"  0    30
run_phase "2-LOW"   10M  60
run_phase "3-IDLE"  0    30
run_phase "4-HIGH"  40M  120
run_phase "5-IDLE"  0    120

#Let controller observe final cooldown
echo "[exp] All phases done. Waiting 30s for controller to stabilise..."
sleep 30

#Stop controller
echo "[exp] Stopping controller (PID $CTRL_PID)..."
kill $CTRL_PID 2>/dev/null
wait $CTRL_PID 2>/dev/null

echo ""
echo "[exp] Done. Results saved to:"
echo "  - $LOG_CSV"
echo "  - $TRAFFIC_LOG"
