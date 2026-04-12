#!/usr/bin/env bash
set -uo pipefail

SERVER="${SERVER_IP:-10.10.6.100}"
PORT="${SERVER_PORT:-5201}"
NS="open5gs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

get_ue_pod() {
  kubectl get pod -n "$NS" -l name=ue1 \
    -o jsonpath='{.items[0].metadata.name}'
}

run_phase() {
  local label="$1" bitrate="$2" dur="$3"
  echo ""
  echo "[burst] $(date -u +%FT%T%z) === Phase: $label | bitrate=$bitrate | dur=${dur}s ==="
  if [ "$bitrate" = "0" ]; then
    echo "[burst] Idle phase — sleeping ${dur}s"
    sleep "$dur"
  else
    for attempt in 1 2 3; do
      UE_POD=$(get_ue_pod)
      echo "[burst] Using UE pod: $UE_POD"
      timeout $((dur + 30)) kubectl exec -n "$NS" "$UE_POD" -c ue -- \
        iperf3 -c "$SERVER" -p "$PORT" -u -b "$bitrate" -l 1200 -t "$dur" \
        --connect-timeout 5000 2>&1 | tail -5
      rc=${PIPESTATUS[0]}
      if [ "$rc" -eq 0 ]; then
        break
      fi
      echo "[burst] iperf3 failed (rc=$rc), restarting RAN and retrying (attempt $attempt/3)..."
      kubectl exec -n "$NS" "$UE_POD" -c ue -- pkill iperf3 2>/dev/null || true
      "$SCRIPT_DIR/../../scripts/restart-ran.sh"
      sleep 5
    done
    sleep 15
  fi
}

echo "========================================="
echo "  Burst Traffic Pattern 1 — Gradual Ramp"
echo "  $(date -u +%FT%T%z)"
echo "========================================="

run_phase "1-IDLE"  0    30
run_phase "2-LOW"   10M  60
run_phase "3-MED"   20M  60
run_phase "4-HIGH"  40M  60
run_phase "5-IDLE"  0    120

echo ""
echo "========================================="
echo "  Burst Traffic Pattern 1 — Complete"
echo "  $(date -u +%FT%T%z)"
echo "========================================="
