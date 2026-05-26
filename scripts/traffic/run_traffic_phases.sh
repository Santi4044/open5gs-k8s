#!/usr/bin/env bash
set -euo pipefail

IDLE1="${IDLE1:-30}"
LOW_RATE="${LOW_RATE:-10M}"
LOW_DUR="${LOW_DUR:-60}"
IDLE2="${IDLE2:-30}"
PEAK_RATE="${PEAK_RATE:-40M}"
PEAK_DUR="${PEAK_DUR:-120}"
IDLE3="${IDLE3:-120}"
OUT_LOG="${OUT_LOG:-/dev/null}"

run_idle_phase() {
  local label="$1" duration="$2"
  echo "$(date -u +%FT%T%z) === Phase: $label | dur=${duration}s ===" | tee -a "$OUT_LOG"
  sleep "$duration"
}

run_traffic_phase() {
  local label="$1" bitrate="$2" duration="$3"
  echo "$(date -u +%FT%T%z) === Phase: $label | bitrate=$bitrate | dur=${duration}s ===" | tee -a "$OUT_LOG"
  BITRATE="$bitrate" DURATION="$duration" scripts/traffic/run_iperf_udp.sh 2>&1 | tee -a "$OUT_LOG"
}

run_idle_phase "1-IDLE" "$IDLE1"
run_traffic_phase "2-LOW" "$LOW_RATE" "$LOW_DUR"
run_idle_phase "3-IDLE" "$IDLE2"
run_traffic_phase "4-HIGH" "$PEAK_RATE" "$PEAK_DUR"
run_idle_phase "5-IDLE" "$IDLE3"
