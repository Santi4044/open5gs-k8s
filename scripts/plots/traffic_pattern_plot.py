import pandas as pd
import matplotlib.pyplot as plt
import os

OUT_DIR = "results/traffic_pattern"
CSV_PATH = f"{OUT_DIR}/watch.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"No traffic data found at {CSV_PATH}. Run scripts/traffic/run_traffic_only.sh first.")

print(f"Using: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Calculate seconds since the first timestamp
df["ts"] = pd.to_datetime(df["ts_iso"], utc=True)
df["elapsed"] = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds()

# Handle NA in pps_prom
df["pps_prom"] = pd.to_numeric(df["pps_prom"], errors="coerce").fillna(0)

# Figure
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
fig.suptitle("Traffic Pattern", fontsize=14, fontweight="bold")

# PPS
ax1.plot(df["elapsed"], df["pps_prom"], color="#2196F3", linewidth=2, label="Actual PPS")
ax1.fill_between(df["elapsed"], df["pps_prom"], alpha=0.1, color="#2196F3")

ax1.set_ylabel("Packets per Second (PPS)", fontsize=11)
ax1.set_xlabel("Time (seconds)", fontsize=11)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

# Phase labels
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

OUT_PNG = f"{OUT_DIR}/traffic_pattern_plot.png"
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
plt.show()

