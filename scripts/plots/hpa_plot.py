import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob
import os

# ── Auto-find latest HPA experiment folder ─────────────────────────────────────
folders = sorted(glob.glob("results/*-hpa-experiment"))
if not folders:
    raise FileNotFoundError("No HPA experiment results found in results/")
latest = folders[-1]
print(f"Using: {latest}")

df = pd.read_csv(f"{latest}/watch.csv")

# ── Compute elapsed seconds from ts_iso ────────────────────────────────────────
df["ts"] = pd.to_datetime(df["ts_iso"], utc=True)
df["elapsed"] = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds()

# ── Handle NA in pps_prom ──────────────────────────────────────────────────────
df["pps_prom"] = pd.to_numeric(df["pps_prom"], errors="coerce").fillna(0)
df["hpa_replicas"] = pd.to_numeric(df["hpa_replicas"], errors="coerce").fillna(1)

# ── Ideal replicas: ceil(pps / threshold), min 1, max 5 ───────────────────────
THRESHOLD = 1500
df["ideal_replicas"] = np.clip(np.ceil(df["pps_prom"] / THRESHOLD), 1, 5).astype(int)

# ── Detect scale events from replica changes ───────────────────────────────────
df["replica_change"] = df["hpa_replicas"].diff()
scale_up   = df[df["replica_change"] > 0]
scale_down = df[df["replica_change"] < 0]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("HPA Autoscaling – Live Experiment", fontsize=14, fontweight="bold")

# ── Top plot: PPS + threshold ──────────────────────────────────────────────────
ax1.plot(df["elapsed"], df["pps_prom"], color="#2196F3", linewidth=2, label="Actual PPS")
ax1.axhline(THRESHOLD, color="#F44336", linewidth=1.5, linestyle="--", label="Threshold (1500 PPS)")
ax1.fill_between(df["elapsed"], df["pps_prom"], alpha=0.1, color="#2196F3")

# ── Scale event markers ────────────────────────────────────────────────────────
ax1.scatter(scale_up["elapsed"],   scale_up["pps_prom"],
            color="green", zorder=5, s=80, marker="^", label="Scale Up")
ax1.scatter(scale_down["elapsed"], scale_down["pps_prom"],
            color="red",   zorder=5, s=80, marker="v", label="Scale Down")

ax1.set_ylabel("Packets per Second (PPS)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

# ── Phase labels ───────────────────────────────────────────────────────────────
phases = [
    (0,   "IDLE\n(30s)"),
    (30,  "LOW\n(10M/60s)"),
    (90,  "IDLE\n(30s)"),
    (120, "HIGH\n(40M/120s)"),
    (240, "IDLE\n(120s)"),
]
for x, label in phases:
    ax1.axvline(x, color="gray", linewidth=0.8, linestyle=":")
    ax1.text(x + 2, ax1.get_ylim()[1] * 0.92, label, fontsize=7.5, color="gray")

# ── Bottom plot: Replicas ──────────────────────────────────────────────────────
ax2.step(df["elapsed"], df["hpa_replicas"], where="post",
         color="#4CAF50", linewidth=2, label="HPA Replicas")
ax2.step(df["elapsed"], df["ideal_replicas"], where="post",
         color="#FF9800", linewidth=1.5, linestyle="--", label="Ideal Replicas")
ax2.set_ylabel("Replicas", fontsize=11)
ax2.set_xlabel("Time (seconds)", fontsize=11)
ax2.set_ylim(0, 6)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Phase lines on bottom plot too ─────────────────────────────────────────────
for x, _ in phases:
    ax2.axvline(x, color="gray", linewidth=0.8, linestyle=":")

plt.tight_layout()
plt.savefig(f"{latest}/hpa_plot.png", dpi=150, bbox_inches="tight")
print(f"Saved: {latest}/hpa_plot.png")
plt.show()
