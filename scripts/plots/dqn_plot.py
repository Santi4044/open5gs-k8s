#!/usr/bin/env python3
"""
DQN Autoscaling Live Experiment — Plot
Mirrors arima_plot.py and hpa_plot.py style for fair visual comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob

# ── Auto-find latest DQN experiment folder ────────────────────────────────────
folders = sorted(glob.glob("results/*-dqn-experiment"))
if not folders:
    raise FileNotFoundError("No DQN experiment results found in results/")
latest = folders[-1]
print(f"Using: {latest}")

df = pd.read_csv(f"{latest}/dqn_live.csv", parse_dates=["ts_iso"])
df = df.sort_values("ts_iso").reset_index(drop=True)

# ── Compute elapsed seconds ───────────────────────────────────────────────────
t0 = df["ts_iso"].iloc[0]
df["elapsed"] = (df["ts_iso"] - t0).dt.total_seconds()

# ── Phase markers ─────────────────────────────────────────────────────────────
phases = [
    (0,   "IDLE\n(30s)"),
    (35,  "LOW\n(10M/60s)"),
    (110, "IDLE\n(30s)"),
    (140, "HIGH\n(40M/120s)"),
    (275, "IDLE\n(120s)"),
]

THRESHOLD = 1500

# ── Ideal replicas ────────────────────────────────────────────────────────────
def ideal_replicas(pps, threshold=THRESHOLD, max_r=5):
    if pps <= 0:
        return 1
    return max(1, min(int(np.ceil(pps / threshold)), max_r))

df["ideal"] = df["pps_actual"].apply(ideal_replicas)

# ── Detect scale events ───────────────────────────────────────────────────────
scale_up   = df[(df["dqn_action"] == "scale up")   & (df["scale_executed"].astype(str).str.strip() == "True")]
scale_down = df[(df["dqn_action"] == "scale down") & (df["scale_executed"].astype(str).str.strip() == "True")]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("DQN Autoscaling – Live Experiment", fontsize=14, fontweight="bold")

# ── Top plot: PPS + threshold ─────────────────────────────────────────────────
ax1.plot(df["elapsed"], df["pps_actual"], color="#2196F3", linewidth=2, label="Actual PPS")
ax1.axhline(THRESHOLD, color="#F44336", linewidth=1.5, linestyle="--", label="Threshold (1500 PPS)")
ax1.fill_between(df["elapsed"], df["pps_actual"], alpha=0.1, color="#2196F3")

ax1.scatter(scale_up["elapsed"],   scale_up["pps_actual"],
            color="#4CAF50", zorder=5, s=80, marker="^", label="Scale Up")
ax1.scatter(scale_down["elapsed"], scale_down["pps_actual"],
            color="#F44336",   zorder=5, s=80, marker="v", label="Scale Down")

ax1.set_ylabel("Packets per Second (PPS)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

# ── Phase labels ──────────────────────────────────────────────────────────────
for x, label in phases:
    ax1.axvline(x, color="gray", linewidth=0.8, linestyle=":")
    ax1.text(x + 2, ax1.get_ylim()[1] * 0.92, label, fontsize=7.5, color="gray")

# ── Bottom plot: DQN Replicas + Ideal Replicas ───────────────────────────────
ax2.step(df["elapsed"], df["current_replicas"], where="post",
         color="#4CAF50", linewidth=2, label="DQN Replicas")
ax2.step(df["elapsed"], df["ideal"], where="post",
         color="#FF9800", linewidth=1.5, linestyle="--", label="Ideal Replicas")
ax2.set_ylabel("Replicas", fontsize=11)
ax2.set_xlabel("Time (seconds)", fontsize=11)
ax2.set_ylim(0, 6)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Phase lines on bottom plot too ────────────────────────────────────────────
for x, _ in phases:
    ax2.axvline(x, color="gray", linewidth=0.8, linestyle=":")

plt.tight_layout()
plt.savefig(f"{latest}/dqn_plot.png", dpi=150, bbox_inches="tight")
print(f"Saved: {latest}/dqn_plot.png")
plt.show()