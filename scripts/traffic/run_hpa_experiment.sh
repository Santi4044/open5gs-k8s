# !/usr/bin/env bash
set -euo pipefail

# Experiment parameters
OUT_DIR="${OUT_DIR:-results/$(date +%Y%m%d-%H%M%S)-hpa-experiment}"
IDLE1="${IDLE1:-30}"
LOW_RATE="${LOW_RATE:-10M}"
LOW_DUR="${LOW_DUR:-60}"
IDLE2="${IDLE2:-30}"
PEAK_RATE="${PEAK_RATE:-40M}"
PEAK_DUR="${PEAK_DUR:-120}"
IDLE3="${IDLE3:-120}"

mkdir -p "$OUT_DIR"

echo "Output dir: $OUT_DIR"

# Start the watcher
OUT_CSV="$OUT_DIR/watch.csv"
OUT_LOG="$OUT_DIR/traffic.log"

OUT_CSV="$OUT_CSV" scripts/traffic/watch_scaling_prom.sh >/dev/null 2>&1 &
WATCH_PID=$!
trap 'kill $WATCH_PID 2>/dev/null || true' EXIT

# Phase 1: idle period
echo "Idle ${IDLE1}s"
sleep "$IDLE1"

# Phase 2: low traffic period
echo "Low traffic ${LOW_RATE} for ${LOW_DUR}s"
BITRATE="$LOW_RATE" DURATION="$LOW_DUR" scripts/traffic/run_iperf_udp.sh | tee -a "$OUT_LOG"

# Phase 3: idle period
echo "Idle ${IDLE2}s"
sleep "$IDLE2"

# Phase 4: high traffic period
echo "Peak traffic ${PEAK_RATE} for ${PEAK_DUR}s (should trigger HPA scale-up)"
BITRATE="$PEAK_RATE" DURATION="$PEAK_DUR" scripts/traffic/run_iperf_udp.sh | tee -a "$OUT_LOG"

# Phase 5: final idle period
echo "Cool-down idle ${IDLE3}s (watching scale down)"
sleep "$IDLE3"

echo "Done. Results saved to:"
echo "  - $OUT_CSV"
echo "  - $OUT_LOG"
