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

# Pick a free local port for the Prometheus port-forward
PROM_LOCAL_PORT="${PROM_LOCAL_PORT:-19090}"
PROM_NAMESPACE="${PROM_NAMESPACE:-monitoring}"
PROM_SVC="${PROM_SVC:-kps-kube-prometheus-stack-prometheus}"

# Start a port-forward to Prometheus in the background
kubectl port-forward -n "$PROM_NAMESPACE" "svc/${PROM_SVC}" \
  "${PROM_LOCAL_PORT}:9090" >/dev/null 2>&1 &
PROM_PF_PID=$!

# Give it a moment to establish
sleep 2

# Update the trap to also kill the port-forward
trap 'kill $WATCH_PID $HPA_PRINT_PID $PROM_PF_PID 2>/dev/null || true' EXIT

hpa_live_print() {
  local prev_replicas=""
  local prom_query='sum(rate(fivegs_ep_n3_gtp_indatapktn3upf{namespace="open5gs",service="open5gs-upf1-metrics"}[30s]))'

  while true; do
    local ts replicas hpa_line action pps

    ts="$(date +%T)"

    # Live PPS from Prometheus
    pps_raw="$(curl -fsS \
      "http://localhost:${PROM_LOCAL_PORT}/api/v1/query" \
      --data-urlencode "query=${prom_query}" 2>/dev/null \
      | sed -n 's/.*"value":\[[^,]*,"\([^"]*\)".*/\1/p' | head -n1)"

    if [[ "$pps_raw" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      pps="$(printf '%.1f' "$pps_raw")"
    else
      pps="NA"
    fi

    hpa_line="$(kubectl get hpa -n "$NAMESPACE_OPEN5GS" "$HPA_NAME" --no-headers 2>/dev/null || true)"
    replicas="$(awk '{print $6}' <<<"$hpa_line" 2>/dev/null || true)"
    replicas="${replicas:-NA}"

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
export HPA_PRINT_PID

trap 'kill $WATCH_PID $HPA_PRINT_PID 2>/dev/null || true' EXIT

# Run traffic
OUT_LOG="$OUT_LOG" scripts/traffic/run_traffic_phases.sh

kill $HPA_PRINT_PID 2>/dev/null || true
echo ""
echo "================================="
echo "Experiment complete: $(date -u +%FT%T%z)"
echo "================================="

echo "Done. Results saved to:"
echo "  - $OUT_CSV"
echo "  - $OUT_LOG"
