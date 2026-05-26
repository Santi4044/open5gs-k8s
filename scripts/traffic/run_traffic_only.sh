# !/usr/bin/env bash
set -uo pipefail

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

OUT_LOG="$OUT_LOG" scripts/traffic/run_traffic_phases.sh

echo ""
echo "Done. Results saved to:"
echo "  - $OUT_CSV"
echo "  - $OUT_LOG"
