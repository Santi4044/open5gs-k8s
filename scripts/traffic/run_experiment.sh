#!/usr/bin/env bash
set -euo pipefail

# Tuning knobs (override via env vars)
OUT_DIR="${OUT_DIR:-results/$(date +%Y%m%d-%H%M%S)-pps-experiment}"
IDLE1="${IDLE1:-30}"
LOW_RATE="${LOW_RATE:-20M}"
LOW_DUR="${LOW_DUR:-60}"
IDLE2="${IDLE2:-30}"
PEAK_RATE="${PEAK_RATE:-150M}"     # adjust if needed to reach 5 replicas
PEAK_DUR="${PEAK_DUR:-120}"
IDLE3="${IDLE3:-180}"

mkdir -p "$OUT_DIR"

echo "[exp] Output dir: $OUT_DIR"

# Start watcher
OUT_CSV="$OUT_DIR/watch.csv"
OUT_LOG="$OUT_DIR/traffic.log"

OUT_CSV="$OUT_CSV" scripts/traffic/watch_scaling_prom.sh >/dev/null 2>&1 &
WATCH_PID=$!
trap 'kill $WATCH_PID 2>/dev/null || true' EXIT

echo "[exp] idle ${IDLE1}s"
sleep "$IDLE1"

echo "[exp] low traffic ${LOW_RATE} for ${LOW_DUR}s"
BITRATE="$LOW_RATE" DURATION="$LOW_DUR" scripts/traffic/run_iperf_udp.sh | tee -a "$OUT_LOG"

echo "[exp] idle ${IDLE2}s"
sleep "$IDLE2"

echo "[exp] peak traffic ${PEAK_RATE} for ${PEAK_DUR}s"
BITRATE="$PEAK_RATE" DURATION="$PEAK_DUR" scripts/traffic/run_iperf_udp.sh | tee -a "$OUT_LOG"

echo "[exp] cool-down idle ${IDLE3}s (watching scale down)"
sleep "$IDLE3"

echo "[exp] done. Logs:"
echo "  - $OUT_CSV"
echo "  - $OUT_LOG"
