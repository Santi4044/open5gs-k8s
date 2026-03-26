#!/usr/bin/env python3
"""Plot DQN autoscaling results for thesis."""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_scaling_comparison(df, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(df["ts"], df["actual_pps"], "b-", linewidth=1.5, label="Actual PPS")
    ax1.axhline(y=4000, color="r", linestyle="--", alpha=0.7, label="HPA Threshold (4000)")
    ax1.set_ylabel("Packets Per Second", fontsize=12)
    ax1.set_title("DQN vs HPA Autoscaling — Traffic & Scaling Decisions", fontsize=14)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2.step(df["ts"], df["actual_replicas"], "r-", linewidth=2, where="post", label="HPA (reactive)")
    ax2.step(df["ts"], df["dqn_replicas"], "g--", linewidth=2, where="post", label="DQN (proactive)")
    ax2.step(df["ts"], df["ideal_replicas"], "k:", linewidth=1.5, where="post", label="Ideal", alpha=0.5)
    ax2.set_ylabel("UPF Replicas", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylim(0.5, max(df["dqn_replicas"].max(), df["actual_replicas"].max()) + 0.5)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.tight_layout()
    path = os.path.join(out_dir, "dqn_scaling_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_training_curve(tc_df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tc_df["episode"], tc_df["total_reward"], "b-", alpha=0.3, linewidth=0.5, label="Per-episode reward")
    window = min(50, len(tc_df)//4) or 1
    tc_df["smooth"] = tc_df["total_reward"].rolling(window=window, min_periods=1).mean()
    ax.plot(tc_df["episode"], tc_df["smooth"], "r-", linewidth=2, label=f"Rolling avg ({window} eps)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("DQN Training Convergence", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "dqn_training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_action_distribution(df, out_dir):
    actions = df["dqn_action"].value_counts()
    colors = {"hold": "#2196F3", "scale_up": "#4CAF50", "scale_down": "#FF9800"}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(actions.index, actions.values,
                  color=[colors.get(a, "#999") for a in actions.index])
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("DQN Action Distribution on Unseen Test Data", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, actions.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "dqn_action_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--training-curve", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.results, parse_dates=["ts"])
    tc = pd.read_csv(args.training_curve)
    print("Generating DQN plots...")
    plot_scaling_comparison(df, args.output_dir)
    plot_training_curve(tc, args.output_dir)
    plot_action_distribution(df, args.output_dir)
    print(f"\nAll plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
