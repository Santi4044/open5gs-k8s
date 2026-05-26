# !/usr/bin/env bash
set -uo pipefail

# Traffic parameters
IDLE1="${IDLE1:-30}"
LOW_RATE="${LOW_RATE:-10M}"
LOW_DUR="${LOW_DUR:-60}"
IDLE2="${IDLE2:-30}"
PEAK_RATE="${PEAK_RATE:-40M}"
PEAK_DUR="${PEAK_DUR:-120}"
IDLE3="${IDLE3:-120}"

OUT_DIR="results/traffic_pattern"
OUT_CSV="${OUT_DIR}/watch.csv"
OUT_LOG="${OUT_DIR}/traffic.log"

mkdir -p "$OUT_DIR"

echo "================================="
echo "Traffic-Only Experiment (No Autoscaling)"
echo "$(date -u +%FT%T%z)"
echo "Output dir: $OUT_DIR"
echo "================================="

# Start watcher
OUT_CSV="$OUT_CSV" scripts/traffic/watch_scaling_prom.sh >/dev/null 2>&1 &
WATCH_PID=$!
trap 'kill $WATCH_PID 2>/dev/null || true' EXIT

echo "Phase 1: IDLE ${IDLE1}s"
sleep "$IDLE1"

echo "Phase 2: LOW traffic ${LOW_RATE} for ${LOW_DUR}s"
BITRATE="$LOW_RATE" DURATION="$LOW_DUR" scripts/traffic/run_iperf_udp.sh | tee -a "$OUT_LOG"

echo "Phase 3: IDLE ${IDLE2}s"
sleep "$IDLE2"

echo "Phase 4: HIGH traffic ${PEAK_RATE} for ${PEAK_DUR}s"
BITRATE="$PEAK_RATE" DURATION="$PEAK_DUR" scripts/traffic/run_iperf_udp.sh | tee -a "$OUT_LOG"

echo "Phase 5: IDLE ${IDLE3}s"
sleep "$IDLE3"

echo ""
echo "Done. Results saved to:"
echo "  - $OUT_CSV"
echo "  - $OUT_LOG"

