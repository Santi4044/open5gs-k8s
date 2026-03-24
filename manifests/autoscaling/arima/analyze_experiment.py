#!/usr/bin/env python3
"""
Post-experiment analysis and visualization.

Reads ARIMA results CSV and generates thesis-quality plots:
  1. PPS over time (actual vs predicted)
  2. Replicas over time (actual HPA vs ARIMA-predicted)
  3. Prediction error over time

Usage:
    python analyze_experiment.py <arima_results.csv> [--save-dir DIR]
"""

import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def plot_pps(df, save_dir):
    """Plot actual vs predicted PPS over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["ts"], df["actual_pps"], label="Actual PPS", color="blue", linewidth=1.5)
    ax.plot(df["ts"], df["predicted_pps"], label="ARIMA Predicted PPS",
            color="red", linewidth=1.5, linestyle="--")
    ax.axhline(y=4000, color="green", linestyle=":", linewidth=1, label="HPA Threshold (4k)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Packets per Second (PPS)")
    ax.set_title("UPF1 N3 GTP-U PPS: Actual vs ARIMA Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, "pps_actual_vs_predicted.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_replicas(df, save_dir):
    """Plot actual HPA replicas vs ARIMA-predicted replicas."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(df["ts"], df["actual_replicas"], label="Actual HPA Replicas",
            color="blue", linewidth=2, where="post")
    ax.step(df["ts"], df["predicted_replicas"], label="ARIMA Predicted Replicas",
            color="red", linewidth=2, linestyle="--", where="post")
    ax.set_xlabel("Time")
    ax.set_ylabel("Replica Count")
    ax.set_title("UPF1 Replicas: Actual HPA vs ARIMA Prediction")
    ax.set_ylim(0, 6)
    ax.set_yticks(range(0, 7))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, "replicas_actual_vs_predicted.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_error(df, save_dir):
    """Plot PPS prediction error over time."""
    df["error"] = df["actual_pps"] - df["predicted_pps"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(df["ts"], df["error"], 0, alpha=0.3, color="orange")
    ax.plot(df["ts"], df["error"], color="orange", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Error (Actual − Predicted PPS)")
    ax.set_title("ARIMA Prediction Error Over Time")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, "prediction_error.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_combined(df, save_dir):
    """Combined 2-panel plot for thesis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: PPS
    ax1.plot(df["ts"], df["actual_pps"], label="Actual PPS", color="blue", linewidth=1.5)
    ax1.plot(df["ts"], df["predicted_pps"], label="ARIMA Predicted",
             color="red", linewidth=1.5, linestyle="--")
    ax1.axhline(y=4000, color="green", linestyle=":", linewidth=1, label="Threshold (4k)")
    ax1.set_ylabel("PPS")
    ax1.set_title("ARIMA-based Predictive Autoscaling Analysis")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: Replicas
    ax2.step(df["ts"], df["actual_replicas"], label="Actual HPA",
             color="blue", linewidth=2, where="post")
    ax2.step(df["ts"], df["predicted_replicas"], label="ARIMA Predicted",
             color="red", linewidth=2, linestyle="--", where="post")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Replicas")
    ax2.set_ylim(0, 6)
    ax2.set_yticks(range(0, 7))
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = os.path.join(save_dir, "combined_analysis.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="ARIMA Experiment Analyzer")
    parser.add_argument("results_csv", help="Path to ARIMA results CSV")
    parser.add_argument("--save-dir", default=None, help="Directory to save plots")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.dirname(args.results_csv) or "."
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading results from {args.results_csv}...")
    df = load_results(args.results_csv)
    print(f"  {len(df)} samples loaded")

    print("\nGenerating plots...")
    plot_pps(df, save_dir)
    plot_replicas(df, save_dir)
    plot_error(df, save_dir)
    plot_combined(df, save_dir)
    print("\nDone! All plots saved to:", save_dir)


if __name__ == "__main__":
    main()
