#!/usr/bin/env bash
set -uo pipefail

DATA_DIR="${1:-$HOME/open5gs-k8s/manifests/autoscaling/dqn/training_data}"
TOTAL_RUNS="${2:-6}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Collecting $TOTAL_RUNS burst experiments"
echo "  Output: $DATA_DIR"
echo "  $(date -u +%FT%T%z)"
echo "============================================"

for i in $(seq 1 $TOTAL_RUNS); do
    if [ "$i" -eq "$TOTAL_RUNS" ]; then
        label="test_1"
        echo ""
        echo ">>> Run $i/$TOTAL_RUNS — TEST SET ($label) <<<"
    else
        label="train_${i}"
        echo ""
        echo ">>> Run $i/$TOTAL_RUNS — TRAINING ($label) <<<"
    fi

    CSV="$DATA_DIR/${label}.csv"

    # Start watcher
    OUT_CSV="$CSV" "$SCRIPT_DIR/watch_scaling_prom.sh" &
    WATCHER_PID=$!
    sleep 3

    # Run burst traffic
    "$SCRIPT_DIR/run_burst_pattern.sh"

    # Wait for trailing data
    sleep 30

    # Stop watcher
    kill $WATCHER_PID 2>/dev/null
    wait $WATCHER_PID 2>/dev/null

    echo "  Saved: $CSV ($(wc -l < "$CSV") lines)"

    # Gap between runs to let iperf3 server fully reset
    if [ "$i" -lt "$TOTAL_RUNS" ]; then
        echo "  Waiting 15s before next run..."
        sleep 15
    fi
done

echo ""
echo "============================================"
echo "  Collection complete!"
echo "  Training: train_1.csv — train_$((TOTAL_RUNS-1)).csv"
echo "  Test:     test_1.csv"
echo "  $(date -u +%FT%T%z)"
echo "============================================"
