#!/usr/bin/env python3
"""
DQN Autoscaling Live Experiment — Plot
Mirrors arima_plot.py style for fair visual comparison.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Load CSV ──────────────────────────────────��──────────────────────
RESULT_FILE = "manifests/autoscaling/dqn/results/dqn_live_experiment.csv"
df = pd.read_csv(RESULT_FILE, parse_dates=["ts_iso"])
df = df.sort_values("ts_iso").reset_index(drop=True)

# ── Trim to row 44 (clean end of IDLE phase) ─────────────────────────
# df = df.iloc[:44].reset_index(drop=True)  # removed hardcoded trim

# ── Compute elapsed seconds ──────────────────────────────────────────
t0 = df["ts_iso"].iloc[0]
df["elapsed"] = (df["ts_iso"] - t0).dt.total_seconds()

#Phase markers
phases = [
    (0, "IDLE\n(30s)"),
    (35, "LOW\n(10M/60s)"),
    (110, "IDLE\n(30s)"),
    (140, "HIGH\n(40M/120s)"),
    (275, "IDLE\n(120s)"),
]

THRESHOLD = 1500

# ── Ideal replicas ───────────────────────────────────────────────────
def ideal_replicas(pps, threshold=THRESHOLD, max_r=5):
    if pps <= 0:
        return 1
    return max(1, min(int(np.ceil(pps / threshold)), max_r))

df["ideal"] = df["pps_actual"].apply(ideal_replicas)

# ── Plot ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("DQN Autoscaling – Live Experiment", fontsize=14, fontweight="bold")

# ── Top: PPS ─────────────────────────────────────────────────────────
ax1.fill_between(df["elapsed"], df["pps_actual"], alpha=0.15, color="steelblue")
ax1.plot(df["elapsed"], df["pps_actual"], color="steelblue", lw=2, label="Actual PPS")
ax1.axhline(THRESHOLD, color="red", linestyle="--", lw=1.5, label=f"Threshold ({THRESHOLD} PPS)")

# Mark scale events
scale_up   = df[(df["dqn_action"] == "scale up")   & (df["scale_executed"].astype(str).str.strip() == "True")]
scale_down = df[(df["dqn_action"] == "scale down") & (df["scale_executed"].astype(str).str.strip() == "True")]
ax1.scatter(scale_up["elapsed"],   scale_up["pps_actual"],
            color="green", zorder=5, s=80, marker="^", label="Scale Up")
ax1.scatter(scale_down["elapsed"], scale_down["pps_actual"],
            color="red",   zorder=5, s=80, marker="v", label="Scale Down")

# Phase lines
for x, label in phases:
    ax1.axvline(x, color="gray", linestyle=":", lw=1)
    ax1.text(x + 1, 4800, label, fontsize=7.5, color="gray", va="top")

ax1.set_ylabel("Packets per Second (PPS)")
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Bottom: Replicas ─────────────────────────────────────────────────
ax2.step(df["elapsed"], df["current_replicas"], where="post",
         color="green", lw=2, label="DQN Replicas")
ax2.step(df["elapsed"], df["ideal"], where="post",
         color="orange", lw=1.5, linestyle="--", label="Ideal Replicas")

for x, label in phases:
    ax2.axvline(x, color="gray", linestyle=":", lw=1)

ax2.set_ylabel("Replicas")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylim(0, 6)
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

out_path = "manifests/autoscaling/dqn/results/dqn_live_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
