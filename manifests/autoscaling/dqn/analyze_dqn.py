#!/usr/bin/env python3
"""
DQN post-experiment analysis and visualization.

Generates:
  1. DQN replicas vs HPA replicas vs Ideal
  2. DQN training reward curve
  3. Action distribution
  4. Combined comparison plot

Usage:
    python analyze_dqn.py <dqn_results.csv> [--training-curve FILE] [--save-dir DIR]
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_replicas(df, save_dir):
    """DQN vs HPA vs Ideal replicas."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(df["ts"], df["actual_replicas"], label="HPA (Actual)",
            color="blue", linewidth=2, where="post")
    ax.step(df["ts"], df["dqn_replicas"], label="DQN Agent",
            color="red", linewidth=2, linestyle="--", where="post")
    ax.step(df["ts"], df["ideal_replicas"], label="Ideal",
            color="green", linewidth=1.5, linestyle=":", where="post")
    ax.set_xlabel("Time")
    ax.set_ylabel("Replica Count")
    ax.set_title("UPF1 Replicas: DQN vs HPA vs Ideal")
    ax.set_ylim(0, 6)
    ax.set_yticks(range(0, 7))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, "dqn_replicas_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_training_curve(curve_path, save_dir):
    """Training reward over episodes."""
    if not curve_path or not os.path.exists(curve_path):
        print("  Skipping training curve (file not found)")
        return
    df = pd.read_csv(curve_path)
    fig, ax = plt.subplots(figsize=(10, 4))

    # Smoothed curve
    window = min(50, len(df) // 5) if len(df) > 10 else 1
    df["smoothed"] = df["total_reward"].rolling(window=window, min_periods=1).mean()

    ax.plot(df["episode"], df["total_reward"], alpha=0.3, color="blue", label="Raw")
    ax.plot(df["episode"], df["smoothed"], color="red", linewidth=2, label=f"Smoothed ({window}-ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "dqn_training_curve.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_actions(df, save_dir):
    """Action distribution pie chart."""
    counts = df["dqn_action"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"hold": "#2ecc71", "scale_up": "#e74c3c", "scale_down": "#3498db"}
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=[colors.get(a, "gray") for a in counts.index],
           startangle=90)
    ax.set_title("DQN Action Distribution")
    plt.tight_layout()
    path = os.path.join(save_dir, "dqn_action_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_reward_over_time(df, save_dir):
    """Reward at each timestep."""
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["green" if r >= 0 else "red" for r in df["reward"]]
    ax.bar(range(len(df)), df["reward"], color=colors, alpha=0.7, width=1.0)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.set_title("DQN Reward per Time Step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "dqn_reward_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="DQN Experiment Analyzer")
    parser.add_argument("results_csv", help="Path to DQN results CSV")
    parser.add_argument("--training-curve", default=None, help="Training curve CSV")
    parser.add_argument("--save-dir", default=None, help="Directory to save plots")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.dirname(args.results_csv) or "."
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading results from {args.results_csv}...")
    df = pd.read_csv(args.results_csv)
    df["ts"] = pd.to_datetime(df["ts"])
    print(f"  {len(df)} samples loaded")

    print("\nGenerating plots...")
    plot_replicas(df, save_dir)
    plot_actions(df, save_dir)
    plot_reward_over_time(df, save_dir)

    # Training curve
    curve = args.training_curve
    if not curve:
        curve = args.results_csv.replace("_results.csv", "_training_curve.csv")
    plot_training_curve(curve, save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
