import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob
import os

# ── Auto-find latest ARIMA experiment folder ───────────────────────────────────
folders = sorted(glob.glob("results/*-arima-experiment"))
if not folders:
    raise FileNotFoundError("No ARIMA experiment results found in results/")
latest = folders[-1]
print(f"Using: {latest}")

df = pd.read_csv(f"{latest}/arima_live.csv")

# ── Compute elapsed seconds from ts_iso ────────────────────────────────────────
df["ts"] = pd.to_datetime(df["ts_iso"], utc=True)
df["elapsed"] = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds()

# ── Clean up columns ───────────────────────────────────────────────────────────
df["pps_actual"]   = pd.to_numeric(df["pps_actual"],   errors="coerce").fillna(0)
df["pps_forecast"] = pd.to_numeric(df["pps_forecast"], errors="coerce").fillna(0)
df["current_replicas"] = pd.to_numeric(df["current_replicas"], errors="coerce").fillna(1)
df["desired_replicas"] = pd.to_numeric(df["desired_replicas"], errors="coerce").fillna(1)

# ── Ideal replicas: ceil(pps / threshold), min 1, max 5 ───────────────────────
THRESHOLD = 1500
df["ideal_replicas"] = np.clip(np.ceil(df["pps_actual"] / THRESHOLD), 1, 5).astype(int)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("ARIMA Autoscaling – Live Experiment", fontsize=14, fontweight="bold")

# ── Top plot: Actual PPS + Forecast PPS + threshold ───────────────────────────
ax1.plot(df["elapsed"], df["pps_actual"],   color="#2196F3", linewidth=2,   label="Actual PPS")
ax1.plot(df["elapsed"], df["pps_forecast"], color="#9C27B0", linewidth=1.5,
         linestyle="--", label="ARIMA Forecast PPS")
ax1.axhline(THRESHOLD, color="#F44336", linewidth=1.5, linestyle="--", label="Threshold (1500 PPS)")
ax1.fill_between(df["elapsed"], df["pps_actual"], alpha=0.1, color="#2196F3")
ax1.set_ylabel("Packets per Second (PPS)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

PHASE_OFFSET = 15

# ── Phase labels ───────────────────────────────────────────────────────────────
phases = [
    (0 + PHASE_OFFSET,   "IDLE\n(30s)"),
    (30 + PHASE_OFFSET,  "LOW\n(10M/60s)"),
    (90 + PHASE_OFFSET,  "IDLE\n(30s)"),
    (120 + PHASE_OFFSET, "HIGH\n(40M/120s)"),
    (240 + PHASE_OFFSET, "IDLE\n(120s)"),
]
for x, label in phases:
    ax1.axvline(x, color="gray", linewidth=0.8, linestyle=":")
    ax1.text(x + 2, ax1.get_ylim()[1] * 0.92, label, fontsize=7.5, color="gray")

# ── Bottom plot: ARIMA Replicas + Ideal Replicas ──────────────────────────────
ax2.step(df["elapsed"], df["current_replicas"], where="post",
         color="#4CAF50", linewidth=2, label="ARIMA Replicas")
ax2.step(df["elapsed"], df["ideal_replicas"], where="post",
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
plt.savefig(f"{latest}/arima_plot.png", dpi=150, bbox_inches="tight")
print(f"Saved: {latest}/arima_plot.png")
plt.show()
