#!/usr/bin/env python3
"""
ARIMA Live Autoscaling Controller for 5G UPF.

Runs a control loop that:
  1. Queries Prometheus for current PPS every --interval seconds
  2. Maintains a sliding window of PPS history
  3. Fits ARIMA on the window and forecasts future PPS
  4. Computes desired replicas from forecasted PPS
  5. Executes kubectl scale if replicas need to change
  6. Logs all decisions to CSV + exposes metrics for Grafana

Usage:
    python arima_live_controller.py --threshold 4000 --interval 5 --horizon 3
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import warnings
from collections import deque
from datetime import datetime, timezone

import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

warnings.filterwarnings("ignore")

#Prometheus query
PROM_URL = os.environ.get(
    "PROM_URL",
    "http://kps-kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
)
PPS_QUERY = (
    'rate(node_netstat_IpExt_InOctets{instance=~".*"}[30s]) or '
    'sum(rate(container_network_receive_packets_total'
    '{namespace="open5gs",pod=~"open5gs-upf1.*"}[30s]))'
)

DEPLOYMENT = "open5gs-upf1"
NAMESPACE  = "open5gs"

running = True

def signal_handler(sig, frame):
    global running
    print("\n[ctrl] Shutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def query_prometheus_via_kubectl():
    """Query Prometheus from inside the cluster using kubectl run (same as watcher)."""
    prom_url = PROM_URL
    query = 'sum(rate(fivegs_ep_n3_gtp_indatapktn3upf{namespace="open5gs",service="open5gs-upf1-metrics"}[30s]))'
    pod_name = f"arima-pq-{int(time.time()) % 100000}"
    try:
        cmd = [
            "kubectl", "run", "-n", "monitoring", "--rm", "-i",
            "--restart=Never", "--quiet", pod_name,
            "--image=curlimages/curl:latest", "--",
            "curl", "-fsS", f"{prom_url}/api/v1/query",
            "--data-urlencode", f"query={query}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            import json
            data = json.loads(result.stdout.strip())
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
    except Exception as e:
        print(f"  [warn] Prometheus query failed: {e}")
    return None


def get_pps_from_prometheus():
    """Get UPF PPS using kubectl-based Prometheus query."""
    pps = query_prometheus_via_kubectl()
    if pps is not None:
        return pps
    return 0.0


def get_current_replicas():
    """Get current replica count from kubectl."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "deploy", DEPLOYMENT, "-n", NAMESPACE,
             "-o", "jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True, timeout=5
        )
        val = result.stdout.strip()
        return int(val) if val else 1
    except Exception:
        return 1


def scale_deployment(desired):
    """Execute kubectl scale."""
    try:
        result = subprocess.run(
            ["kubectl", "scale", "deploy", DEPLOYMENT, "-n", NAMESPACE,
             f"--replicas={desired}"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  [error] Scale failed: {e}")
        return False


def pps_to_replicas(pps, threshold=4000, max_replicas=5):
    """Convert PPS to desired replica count."""
    if pps <= 0:
        return 1
    desired = int(np.ceil(pps / threshold))
    return max(1, min(desired, max_replicas))


def arima_forecast(history, order=(2, 1, 2), horizon=3):
    """Fit ARIMA on history and return forecasted PPS."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(list(history), order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=horizon)
        return max(0, float(np.mean(forecast)))
    except Exception:
        # Fallback: exponential moving average
        arr = np.array(list(history))
        weights = np.exp(np.linspace(-1, 0, len(arr)))
        weights /= weights.sum()
        return max(0, float(np.dot(arr, weights)))


def main():
    parser = argparse.ArgumentParser(description="ARIMA Live Controller")
    parser.add_argument("--threshold", type=int, default=1500,
                        help="PPS per replica (default: 1,500)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Control loop interval in seconds (default: 5)")
    parser.add_argument("--horizon", type=int, default=3,
                        help="ARIMA forecast horizon (default: 3)")
    parser.add_argument("--order", nargs=3, type=int, default=[2, 1, 2],
                        metavar=("P", "D", "Q"))
    parser.add_argument("--window", type=int, default=30,
                        help="Sliding window size for ARIMA (default: 30)")
    parser.add_argument("--min-window", type=int, default=10,
                        help="Min samples before ARIMA starts (default: 10)")
    parser.add_argument("--max-replicas", type=int, default=5)
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Cooldown seconds between scale actions (default: 30)")
    parser.add_argument("--log", type=str,
                        default="manifests/autoscaling/arima/results/arima_live_log.csv",
                        help="Output log CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually scale, just log decisions")
    args = parser.parse_args()

    order = tuple(args.order)
    history = deque(maxlen=args.window)
    last_scale_time = 0

    # Setup CSV log
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log_file = open(args.log, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "ts_iso", "pps_actual", "pps_forecast", "current_replicas",
        "desired_replicas", "action", "scale_executed"
    ])

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"""
=======================================================
ARIMA Live Autoscaling Controller
Mode:       {mode:<35s}
ARIMA:      {str(order):<35s}
Horizon:    {args.horizon:<35d}
Threshold:  {args.threshold:<35d}
Interval:   {args.interval}s{'':<33s}
Window:     {args.window:<35d}
Cooldown:   {args.cooldown}s{'':<33s}
Log:        {os.path.basename(args.log):<35s}
=======================================================
    """)

    if not args.dry_run:
        # Disable HPA so ARIMA controls scaling
        print("[init] Disabling HPA for UPF1...")
        subprocess.run(
            ["kubectl", "delete", "hpa", "open5gs-upf1-pps", "-n", NAMESPACE],
            capture_output=True, text=True
        )
        print("[init] HPA disabled. ARIMA is now in control.\n")

    step = 0
    try:
        while running:
            loop_start = time.time()
            now = datetime.now(timezone.utc)
            ts = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")

            # 1. Get current PPS
            pps = get_pps_from_prometheus()
            history.append(pps)
            current_replicas = get_current_replicas()

            # 2. Forecast
            if len(history) >= args.min_window:
                forecast_pps = arima_forecast(history, order, args.horizon)
            else:
                forecast_pps = pps  # not enough data yet, use current

            # 3. Decide replicas
            desired = pps_to_replicas(forecast_pps, args.threshold, args.max_replicas)

            # 4. Determine action
            action = "hold"
            scaled = False
            time_since_last = time.time() - last_scale_time

            if desired != current_replicas and time_since_last >= args.cooldown:
                action = "scale up" if desired > current_replicas else "scale down"
                if not args.dry_run:
                    scaled = scale_deployment(desired)
                    if scaled:
                        last_scale_time = time.time()
                else:
                    scaled = False  # dry run

            # 5. Log
            writer.writerow([ts, round(pps, 2), round(forecast_pps, 2),
                             current_replicas, desired, action, scaled])
            log_file.flush()

            # 6. Print status
            step += 1
            print(f"  [{step:>4d}] {ts} | PPS: {pps:>8.1f} | "
                  f"Forecast: {forecast_pps:>8.1f} | "
                  f"Replicas: {current_replicas} -> {desired} | "
                  f"{action:<10s}")

            elapsed = time.time() - loop_start
            time.sleep(max(0, args.interval - elapsed))

    finally:
        log_file.close()
        print(f"\n[done] Log saved to {args.log}")
        print(f"[done] {step} control loop iterations completed")

        if not args.dry_run:
            print("[cleanup] Re-enabling HPA...")
            subprocess.run(
                ["kubectl", "apply", "-f",
                 "manifests/autoscaling/hpa-upf1-pps.yaml"],
                capture_output=True, text=True
            )
            print("[cleanup] HPA re-enabled.")


if __name__ == "__main__":
    main()
