# !/usr/bin/env bash
set -uo pipefail

# Create a timestamped output folder
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="results/${TIMESTAMP}-arima-experiment"
mkdir -p "$OUT_DIR"
echo "[exp] Output dir: $OUT_DIR"

LOG_CSV="${OUT_DIR}/arima_live.csv"
TRAFFIC_LOG="${OUT_DIR}/traffic.log"

# Start the ARIMA live controller
echo "Starting ARIMA live controller..."
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

# Wait for the controller to initialise
echo "Waiting 3s for the controller to initialise..."

sleep 3

OUT_LOG="$TRAFFIC_LOG" scripts/traffic/run_traffic_phases.sh

# Wait before stopping the controller
echo "All traffic phases done. Waiting 30s for controller to stabilise..."
sleep 30

# Stop the controller
echo "Stopping controller (PID $CTRL_PID)..."
kill $CTRL_PID 2>/dev/null
wait $CTRL_PID 2>/dev/null

echo ""
echo "Done. Results saved to:"
echo "  - $LOG_CSV"
echo "  - $TRAFFIC_LOG"
