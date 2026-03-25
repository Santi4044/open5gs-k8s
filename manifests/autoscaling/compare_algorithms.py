#!/usr/bin/env python3
"""
Final comparison: HPA vs ARIMA vs DQN

Generates a unified comparison table and combined plot
for thesis presentation.

Usage:
    python compare_algorithms.py <experiment_name>
    Example: python compare_algorithms.py exp_burst
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_all(exp_name):
    """Load raw CSV, ARIMA results, and DQN results for an experiment."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    raw = pd.read_csv(f"arima/results/{exp_name}.csv")
    raw["ts"] = pd.to_datetime(raw["ts_iso"])
    raw["pps"] = pd.to_numeric(raw["pps_prom"], errors="coerce").fillna(0)
    raw["replicas"] = pd.to_numeric(raw["hpa_replicas"], errors="coerce").fillna(1).astype(int)

    arima = pd.read_csv(f"arima/results/{exp_name}_arima_results.csv")
    arima["ts"] = pd.to_datetime(arima["ts"])

    dqn = pd.read_csv(f"dqn/results/{exp_name}_dqn_results.csv")
    dqn["ts"] = pd.to_datetime(dqn["ts"])

    return raw, arima, dqn


def compute_comparison(raw, arima, dqn, threshold=4000):
    """Compute comparison metrics for all three algorithms."""
    
    # HPA metrics (from raw data)
    hpa_replicas = raw["replicas"].values
    hpa_scale_events = int((pd.Series(hpa_replicas).diff().fillna(0) != 0).sum())
    
    # Ideal replicas
    ideal = np.array([max(1, min(int(np.ceil(p / threshold)), 5)) 
                       if p > 0 else 1 for p in raw["pps"].values])
    
    hpa_match_ideal = np.mean(hpa_replicas[:len(ideal)] == ideal[:len(hpa_replicas)]) * 100

    # ARIMA metrics
    arima_match = arima["replica_match"].mean() * 100 if "replica_match" in arima else 0
    arima_replicas = arima["predicted_replicas"].values
    arima_scale_events = int((pd.Series(arima_replicas).diff().fillna(0) != 0).sum())

    # DQN metrics
    dqn_match_hpa = dqn["replica_match_hpa"].mean() * 100
    dqn_match_ideal = dqn["replica_match_ideal"].mean() * 100
    dqn_replicas = dqn["dqn_replicas"].values
    dqn_scale_events = int((pd.Series(dqn_replicas).diff().fillna(0) != 0).sum())

    return {
        "HPA": {
            "match_vs_ideal": round(hpa_match_ideal, 1),
            "scale_events": hpa_scale_events,
            "type": "Reactive",
        },
        "ARIMA": {
            "match_vs_ideal": round(arima_match, 1),
            "scale_events": arima_scale_events,
            "type": "Predictive",
        },
        "DQN": {
            "match_vs_ideal": round(dqn_match_ideal, 1),
            "scale_events": dqn_scale_events,
            "type": "RL-based",
        },
    }


def plot_combined(raw, arima, dqn, exp_name, save_dir, threshold=4000):
    """Generate combined thesis comparison plot."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # ── Panel 1: PPS Traffic ──
    ax = axes[0]
    ax.plot(raw["ts"], raw["pps"], color="black", linewidth=1.5, label="Actual PPS")
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label=f"HPA Threshold ({threshold})")
    ax.set_ylabel("Packets per Second")
    ax.set_title(f"Algorithm Comparison — {exp_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Replica Comparison ──
    ax = axes[1]
    # Ideal
    ideal = [max(1, min(int(np.ceil(p / threshold)), 5)) if p > 0 else 1 
             for p in raw["pps"].values]
    ax.step(raw["ts"], ideal, color="green", linewidth=1.5, linestyle=":",
            label="Ideal", where="post", alpha=0.7)
    ax.step(raw["ts"], raw["replicas"], color="blue", linewidth=2,
            label="HPA (Reactive)", where="post")
    
    # Align ARIMA and DQN timestamps
    if len(arima) > 0:
        ax.step(arima["ts"], arima["predicted_replicas"], color="orange",
                linewidth=2, linestyle="--", label="ARIMA (Predictive)", where="post")
    if len(dqn) > 0:
        ax.step(dqn["ts"], dqn["dqn_replicas"], color="red",
                linewidth=2, linestyle="-.", label="DQN (RL-based)", where="post")

    ax.set_ylabel("Replica Count")
    ax.set_ylim(0, 6)
    ax.set_yticks(range(0, 7))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Accuracy bars ──
    ax = axes[2]
    # Show match vs ideal as bar chart at each timestep for DQN
    if "reward" in dqn.columns:
        colors = ["green" if r >= 0 else "red" for r in dqn["reward"]]
        ax.bar(range(len(dqn)), dqn["reward"], color=colors, alpha=0.6, width=1.0)
        ax.set_ylabel("DQN Reward")
        ax.set_xlabel("Time Step")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"comparison_{exp_name}.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare HPA vs ARIMA vs DQN")
    parser.add_argument("experiment", help="Experiment name (e.g., exp_burst)")
    parser.add_argument("--save-dir", default="comparison_results",
                        help="Output directory")
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {args.experiment} data...")
    raw, arima, dqn = load_all(args.experiment)
    print(f"  Raw: {len(raw)} samples")
    print(f"  ARIMA: {len(arima)} samples")
    print(f"  DQN: {len(dqn)} samples")

    print(f"\nComputing comparison metrics...")
    metrics = compute_comparison(raw, arima, dqn)

    print(f"\n{'='*60}")
    print(f"  ALGORITHM COMPARISON — {args.experiment}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'HPA':>10} {'ARIMA':>10} {'DQN':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Type':<25} {'Reactive':>10} {'Predictive':>10} {'RL-based':>10}")
    print(f"  {'Match vs Ideal (%)':<25} {metrics['HPA']['match_vs_ideal']:>10} {metrics['ARIMA']['match_vs_ideal']:>10} {metrics['DQN']['match_vs_ideal']:>10}")
    print(f"  {'Scale Events':<25} {metrics['HPA']['scale_events']:>10} {metrics['ARIMA']['scale_events']:>10} {metrics['DQN']['scale_events']:>10}")
    print(f"{'='*60}")

    print(f"\nGenerating comparison plot...")
    plot_combined(raw, arima, dqn, args.experiment, save_dir)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics).T
    metrics_path = os.path.join(save_dir, f"metrics_{args.experiment}.csv")
    metrics_df.to_csv(metrics_path)
    print(f"  Saved: {metrics_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
