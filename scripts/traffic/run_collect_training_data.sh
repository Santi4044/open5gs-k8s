# !/usr/bin/env bash
set -uo pipefail

TRAIN_DIR="manifests/autoscaling/dqn/training_data"
mkdir -p "$TRAIN_DIR"

echo "=============================="
echo "DQN Training Data Collection"
echo "$(date -u +%FT%T%z)"
echo "=============================="

for i in 1 2 3; do
    echo ""
    echo "--- Collecting Pattern $i ---"

    # Start watcher
    OUT_CSV="$TRAIN_DIR/train_${i}.csv" bash scripts/traffic/watch_scaling_prom.sh &
    WATCHER_PID=$!

    # Run traffic pattern
    bash scripts/traffic/run_pattern${i}.sh

    # Stop watcher
    kill $WATCHER_PID 2>/dev/null
    wait $WATCHER_PID 2>/dev/null
    echo "Saved: $TRAIN_DIR/train_${i}.csv"

    # Small break between patterns
    sleep 10
done

echo ""
echo "=============================="
echo "All training data collected"
echo "$(date -u +%FT%T%z)"
echo "=============================="
echo ""
ls -lh "$TRAIN_DIR/"
