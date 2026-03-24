#!/usr/bin/env python3
"""
ARIMA-based PPS Predictor for 5G UPF Autoscaling.

Reads watcher CSV (from watch_scaling_prom.sh), fits an ARIMA model
on the pps_prom time series, and predicts future PPS values.
From predicted PPS, derives a "predicted desired replica count"
and compares it to actual HPA decisions.

Usage:
    python arima_predictor.py <csv_file> [--order P D Q] [--horizon N] [--threshold T]
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def load_data(csv_path):
    """Load watcher CSV and parse timestamps."""
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts_iso"])
    df = df.set_index("ts").sort_index()
    df["pps"] = pd.to_numeric(df["pps_prom"], errors="coerce").fillna(0)
    df["replicas"] = pd.to_numeric(df["hpa_replicas"], errors="coerce").fillna(1).astype(int)
    return df


def check_stationarity(series):
    """Run ADF test and return (is_stationary, p_value)."""
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1] < 0.05, result[1]


def pps_to_replicas(pps, threshold=4000, max_replicas=5):
    """Convert PPS value to desired replica count (mirrors HPA logic)."""
    if pps <= 0:
        return 1
    desired = int(np.ceil(pps / threshold))
    return max(1, min(desired, max_replicas))


def rolling_arima_forecast(df, order=(2, 1, 2), horizon=3, threshold=4000,
                           min_window=10):
    """
    Perform rolling ARIMA forecast:
    - At each time step t, fit ARIMA on pps[0:t]
    - Predict next `horizon` steps
    - Record predicted PPS and derived replica count
    """
    pps = df["pps"].values
    n = len(pps)

    results = []
    for t in range(min_window, n):
        train = pps[:t]

        try:
            model = ARIMA(train, order=order)
            fit = model.fit()
            forecast = fit.forecast(steps=horizon)
            pred_pps = max(0, forecast[0])  # next-step prediction
            pred_pps_avg = max(0, np.mean(forecast))  # avg over horizon
        except Exception:
            pred_pps = train[-1]  # fallback: persist last value
            pred_pps_avg = train[-1]

        pred_replicas = pps_to_replicas(pred_pps_avg, threshold)
        actual_pps = pps[t] if t < n else np.nan
        actual_replicas = df["replicas"].iloc[t] if t < n else np.nan

        results.append({
            "ts": df.index[t],
            "actual_pps": actual_pps,
            "predicted_pps": round(pred_pps, 2),
            "predicted_pps_avg": round(pred_pps_avg, 2),
            "actual_replicas": actual_replicas,
            "predicted_replicas": pred_replicas,
            "replica_match": pred_replicas == actual_replicas,
        })

    return pd.DataFrame(results)


def evaluate(results_df):
    """Compute evaluation metrics."""
    valid = results_df.dropna(subset=["actual_pps", "predicted_pps"])

    # PPS prediction error
    mae = np.mean(np.abs(valid["actual_pps"] - valid["predicted_pps"]))
    rmse = np.sqrt(np.mean((valid["actual_pps"] - valid["predicted_pps"]) ** 2))

    # Replica accuracy
    match_rate = valid["replica_match"].mean() * 100

    # Scaling reaction comparison
    actual_changes = (valid["actual_replicas"].diff().fillna(0) != 0).sum()
    predicted_changes = (valid["predicted_replicas"].diff().fillna(0) != 0).sum()

    return {
        "mae_pps": round(mae, 2),
        "rmse_pps": round(rmse, 2),
        "replica_match_pct": round(match_rate, 1),
        "actual_scale_events": int(actual_changes),
        "predicted_scale_events": int(predicted_changes),
        "total_samples": len(valid),
    }


def main():
    parser = argparse.ArgumentParser(description="ARIMA PPS Predictor for 5G UPF")
    parser.add_argument("csv_file", help="Path to watcher CSV")
    parser.add_argument("--order", nargs=3, type=int, default=[2, 1, 2],
                        metavar=("P", "D", "Q"), help="ARIMA(p,d,q) order (default: 2 1 2)")
    parser.add_argument("--horizon", type=int, default=3,
                        help="Forecast horizon in steps (default: 3)")
    parser.add_argument("--threshold", type=int, default=4000,
                        help="PPS threshold per replica (default: 4000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path for results")
    args = parser.parse_args()

    print(f"Loading data from {args.csv_file}...")
    df = load_data(args.csv_file)
    print(f"  Loaded {len(df)} samples from {df.index[0]} to {df.index[-1]}")
    print(f"  PPS range: {df['pps'].min():.1f} — {df['pps'].max():.1f}")
    print(f"  Replica range: {df['replicas'].min()} — {df['replicas'].max()}")

    # Stationarity check
    is_stationary, p_val = check_stationarity(df["pps"])
    print(f"\n  ADF test: p={p_val:.4f} → {'stationary' if is_stationary else 'non-stationary'}")

    order = tuple(args.order)
    print(f"\nRunning rolling ARIMA{order} forecast (horizon={args.horizon})...")
    results = rolling_arima_forecast(df, order=order, horizon=args.horizon,
                                     threshold=args.threshold)

    metrics = evaluate(results)
    print(f"\n{'='*50}")
    print(f"  ARIMA{order} Evaluation Results")
    print(f"{'='*50}")
    print(f"  Samples evaluated:       {metrics['total_samples']}")
    print(f"  PPS MAE:                 {metrics['mae_pps']}")
    print(f"  PPS RMSE:                {metrics['rmse_pps']}")
    print(f"  Replica match rate:      {metrics['replica_match_pct']}%")
    print(f"  Actual scale events:     {metrics['actual_scale_events']}")
    print(f"  Predicted scale events:  {metrics['predicted_scale_events']}")

    # Save results
    out_path = args.output or args.csv_file.replace(".csv", "_arima_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    return metrics


if __name__ == "__main__":
    main()
