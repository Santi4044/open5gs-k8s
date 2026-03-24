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
    # Retry up to 3 times if iperf3 server is busy
    for attempt in 1 2 3; do
      kubectl exec -n "$NS" "$UE_POD" -c ue -- \
        iperf3 -c "$SERVER" -p "$PORT" -u -b "$bitrate" -l 1200 -t "$dur" \
        --connect-timeout 5000 2>&1 | tail -5
      if [ ${PIPESTATUS[0]} -eq 0 ]; then
        break
      fi
      echo "[burst] iperf3 busy, retrying in 10s (attempt $attempt/3)..."
      sleep 10
    done
    sleep 10  # longer gap to let server fully release
  fi
}

echo "========================================="
echo "  Burst Traffic Pattern — Start"
echo "  $(date -u +%FT%T%z)"
echo "========================================="

run_phase "1-LOW"   20M  30
run_phase "2-HIGH"  50M  30
run_phase "3-LOW"   20M  30
run_phase "4-HIGH"  50M  30
run_phase "5-IDLE"  0    30

echo ""
echo "========================================="
echo "  Burst Traffic Pattern — Complete"
echo "  $(date -u +%FT%T%z)"
echo "========================================="
