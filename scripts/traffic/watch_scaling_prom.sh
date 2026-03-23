#!/usr/bin/env bash
set -euo pipefail
exec </dev/null

NAMESPACE_OPEN5GS="${NAMESPACE_OPEN5GS:-open5gs}"
HPA_NAME="${HPA_NAME:-open5gs-upf1-pps}"
DEPLOY_NAME="${DEPLOY_NAME:-open5gs-upf1}"

PROM_URL="${PROM_URL:-http://kps-kube-prometheus-stack-prometheus.monitoring.svc:9090}"
PROM_NAMESPACE="${PROM_NAMESPACE:-open5gs}"
PROM_INTERVAL_SECONDS="${PROM_INTERVAL_SECONDS:-2}"
OUT_CSV="${OUT_CSV:-results/watch.csv}"

# UPF1-only (sum across replicas)
PROMQL="${PROMQL:-sum(rate(fivegs_ep_n3_gtp_indatapktn3upf{namespace=\"${PROM_NAMESPACE}\",service=\"open5gs-upf1-metrics\"}[30s]))}"

mkdir -p "$(dirname "$OUT_CSV")"
echo "ts_iso,pps_prom,hpa_replicas,hpa_current,hpa_target,deploy_ready,deploy_replicas" > "$OUT_CSV"

trap 'exit 0' INT TERM

# query Prometheus from inside cluster using a short-lived curl pod
prom_query() {
  local q="$1"

  local enc
  enc="$(Q="$q" python3 - <<'PY'
import os, urllib.parse
print(urllib.parse.quote(os.environ["Q"], safe=""))
PY
)"

  # unique pod name each call to avoid collisions
  kubectl run -n monitoring --rm -i --restart=Never --quiet "promq-$RANDOM" --image=curlimages/curl:8.5.0 -- \
    sh -lc "curl -fsS '${PROM_URL}/api/v1/query?query=${enc}'" 2>/dev/null \
  | sed -n 's/.*"value":\[[^,]*,"\([^"]*\)".*/\1/p' | head -n1
}

while true; do
  ts="$(date -Iseconds)"

  pps="$(prom_query "$PROMQL" || true)"
  pps="${pps:-NA}"

  hpa_line="$(kubectl get hpa -n "$NAMESPACE_OPEN5GS" "$HPA_NAME" --no-headers 2>/dev/null || true)"
  hpa_repl="$(awk '{print $6}' <<<"$hpa_line" 2>/dev/null || true)"
  hpa_targets="$(awk '{print $3}' <<<"$hpa_line" 2>/dev/null || true)"
  hpa_current="${hpa_targets%/*}"
  hpa_target="${hpa_targets#*/}"

  dep_line="$(kubectl get deploy -n "$NAMESPACE_OPEN5GS" "$DEPLOY_NAME" --no-headers 2>/dev/null || true)"
  dep_ready="$(awk '{print $2}' <<<"$dep_line" 2>/dev/null || true)"
  dep_repl="$(awk '{print $4}' <<<"$dep_line" 2>/dev/null || true)"

  echo "${ts},${pps},${hpa_repl:-NA},${hpa_current:-NA},${hpa_target:-NA},${dep_ready:-NA},${dep_repl:-NA}" >> "$OUT_CSV"
  sleep "$PROM_INTERVAL_SECONDS"
done
