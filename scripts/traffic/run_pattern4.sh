#!/usr/bin/env bash
set -uo pipefail

SERVER="${SERVER_IP:-10.10.6.100}"
PORT="${SERVER_PORT:-5201}"
NS="open5gs"

UE_POD=$(kubectl get pod -n "$NS" -l name=ue1 \
         -o jsonpath='{.items[0].metadata.name}')

run_phase() {
  local label="$1" bitrate="$2" dur="$3"
  echo ""
  echo "[burst] $(date -u +%FT%T%z) === Phase: $label | bitrate=$bitrate | dur=${dur}s ==="
  if [ "$bitrate" = "0" ]; then
    echo "[burst] Idle phase — sleeping ${dur}s"
    sleep "$dur"
  else
    for attempt in 1 2 3; do
      timeout $((dur + 30)) kubectl exec -n "$NS" "$UE_POD" -c ue -- \
        iperf3 -c "$SERVER" -p "$PORT" -u -b "$bitrate" -l 1200 -t "$dur" \
        --connect-timeout 5000 2>&1 | tail -5
      rc=${PIPESTATUS[0]}
      if [ "$rc" -eq 0 ]; then
        break
      fi
      echo "[burst] iperf3 failed (rc=$rc), retrying in 15s (attempt $attempt/3)..."
      kubectl exec -n "$NS" "$UE_POD" -c ue -- pkill iperf3 2>/dev/null
      sleep 15
    done
    sleep 15  # gap between phases
  fi
}

echo "========================================="
echo "  Burst Traffic Pattern — Start"
echo "  $(date -u +%FT%T%z)"
echo "========================================="

# ── CHANGE THESE PHASES PER RUN ──────────────────────────
run_phase "1-IDLE"  0    30
run_phase "2-LOW"   10M  60
run_phase "3-IDLE"  0    30
run_phase "4-HIGH"  40M  90
run_phase "5-IDLE"  0    120
# ─────────────────────────────────────────────────────────

echo ""
echo "========================================="
echo "  Burst Traffic Pattern — Complete"
echo "  $(date -u +%FT%T%z)"
echo "========================================="
