# !/usr/bin/env bash
set -euo pipefail

# Traffic phase parameters
IDLE1="${IDLE1:-30}"
LOW_RATE="${LOW_RATE:-10M}"
LOW_DUR="${LOW_DUR:-60}"
IDLE2="${IDLE2:-30}"
PEAK_RATE="${PEAK_RATE:-40M}"
PEAK_DUR="${PEAK_DUR:-120}"
IDLE3="${IDLE3:-120}"
OUT_LOG="${OUT_LOG:-/dev/null}"

print_phase_banner() {
  # Pause HPA printer so the banner isn't interleaved
  [[ -n "${HPA_PRINT_PID:-}" ]] && kill -STOP "$HPA_PRINT_PID" 2>/dev/null || true
  sleep 0.3
  echo ""
  echo "$(date -u +%FT%T%z) === Phase: $1 ==="
  [[ -n "${HPA_PRINT_PID:-}" ]] && kill -CONT "$HPA_PRINT_PID" 2>/dev/null || true
}

# Sleep for a given duration and log the phase label
run_idle_phase() {
  local label="$1" duration="$2"
  print_phase_banner "$label | dur=${duration}s"
  sleep "$duration"
}

# Send UDP traffic via iperf3 and log the phase label
run_traffic_phase() {
  local label="$1" bitrate="$2" duration="$3"
  print_phase_banner "$label | bitrate=$bitrate | dur=${duration}s"
  BITRATE="$bitrate" DURATION="$duration" scripts/traffic/run_iperf_udp.sh >> "$OUT_LOG" 2>&1
}

# Run five-phase traffic pattern
run_idle_phase "1-IDLE" "$IDLE1"
run_traffic_phase "2-LOW" "$LOW_RATE" "$LOW_DUR"
run_idle_phase "3-IDLE" "$IDLE2"
run_traffic_phase "4-HIGH" "$PEAK_RATE" "$PEAK_DUR"
run_idle_phase "5-IDLE" "$IDLE3"
