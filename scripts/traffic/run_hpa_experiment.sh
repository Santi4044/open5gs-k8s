# !/usr/bin/env bash
set -euo pipefail

# Experiment parameters
OUT_DIR="${OUT_DIR:-results/$(date +%Y%m%d-%H%M%S)-hpa-experiment}"
NAMESPACE_OPEN5GS="${NAMESPACE_OPEN5GS:-open5gs}"
HPA_NAME="${HPA_NAME:-open5gs-upf1-pps}"

mkdir -p "$OUT_DIR"

echo "================================="
echo "HPA Live Scaling Experiment"
echo "$(date -u +%FT%T%z)"
echo "Output dir: $OUT_DIR"
echo "================================="

# Start the watcher
OUT_CSV="$OUT_DIR/watch.csv"
OUT_LOG="$OUT_DIR/traffic.log"

OUT_CSV="$OUT_CSV" scripts/traffic/watch_scaling_prom.sh >/dev/null 2>&1 &
WATCH_PID=$!

hpa_live_print() {
  local prev_replicas=""
  while true; do
    local ts hpa_line replicas targets pps action
    ts="$(date +%T)"
    hpa_line="$(kubectl get hpa -n "$NAMESPACE_OPEN5GS" "$HPA_NAME" --no-headers 2>/dev/null || true)"
    replicas="$(awk '{print $6}' <<<"$hpa_line" 2>/dev/null || true)"
    targets="$(awk '{print $3}' <<<"$hpa_line" 2>/dev/null || true)"
    pps_raw="${targets%/*}"
    if [[ "$pps_raw" =~ ^([0-9]+)m$ ]]; then
      pps=$(echo "scale=1; ${BASH_REMATCH[1]} / 1000" | bc)
    else
      pps="$pps_raw"
    fi
    replicas="${replicas:-NA}"
    pps="${pps:-NA}"

    action="Hold"
    if [[ "$replicas" =~ ^[0-9]+$ && "$prev_replicas" =~ ^[0-9]+$ ]]; then
      if (( replicas > prev_replicas )); then
        action="Scale Up"
      elif (( replicas < prev_replicas )); then
        action="Scale Down"
      fi
    fi

    echo "[HPA] ${ts} | PPS: ${pps} | Replicas: ${replicas} | Action: ${action}"
    prev_replicas="$replicas"
    sleep 5
  done
}

hpa_live_print &
HPA_PRINT_PID=$!

trap 'kill $WATCH_PID $HPA_PRINT_PID 2>/dev/null || true' EXIT

# Run traffic
OUT_LOG="$OUT_LOG" scripts/traffic/run_traffic_phases.sh

echo "Done. Results saved to:"
echo "  - $OUT_CSV"
echo "  - $OUT_LOG"
