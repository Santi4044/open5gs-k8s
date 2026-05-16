# !/usr/bin/env python3
"""
DQN Training Plots - Reward Function and Epsilon
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Auto-find the latest DQN experiment folder that has the training curve CSV
folders = sorted(glob.glob("results/*-dqn-experiment"), reverse=True)
CSV = None
OUT_DIR = None
for folder in folders:
    candidate = os.path.join(folder, "dqn_live_training_curve.csv")
    if os.path.isfile(candidate):
        CSV = candidate
        OUT_DIR = folder
        break

if CSV is None:
    raise FileNotFoundError("No dqn_live_training_curve.csv found in any results/*-dqn-experiment folder")

print(f"Using: {OUT_DIR}")

df = pd.read_csv(CSV)

WINDOW = 50

# Calculate epsilon values
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min   = 0.01
df["epsilon"] = df["episode"].apply(
    lambda e: max(epsilon_min, epsilon_start * (epsilon_decay ** e))
)

# Calculate rolling average reward
df["rolling_avg"] = df["total_reward"].rolling(window=WINDOW, min_periods=1).mean()


# === Plot 1 - Reward Function Plot ===
fig1, ax1 = plt.subplots(figsize=(12, 5))
fig1.suptitle("DQN Training - Reward per Episode", fontsize=14, fontweight="bold")

ax1.plot(df["episode"], df["rolling_avg"],
         color="darkorange", lw=2.5, label=f"Rolling Average (window={WINDOW})")
ax1.axhline(df["rolling_avg"].iloc[-1], color="green", linestyle="--", lw=1.5,
            label=f"Final Avg: {df['rolling_avg'].iloc[-1]:.0f}")

ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# === Plot 2 - Epsilon Decay Plot ===
fig2, ax2 = plt.subplots(figsize=(12, 4))
fig2.suptitle("DQN Training – Epsilon Decay (Exploration -> Exploitation)",
              fontsize=14, fontweight="bold")

ax2.plot(df["episode"], df["epsilon"], color="crimson", lw=2.5, label="Epsilon")
ax2.axhline(epsilon_min, color="gray", linestyle="--", lw=1.5,
            label=f"Min Epsilon = {epsilon_min}")

ep_floor = df[df["epsilon"] <= epsilon_min + 0.001]["episode"].iloc[0]
ax2.axvspan(0, ep_floor, alpha=0.05, color="red", label="Exploration")
ax2.axvspan(ep_floor, df["episode"].max(), alpha=0.05, color="green", label="Exploitation")

ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

out1 = f"{OUT_DIR}/dqn_reward_curve.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")

out2 = f"{OUT_DIR}/dqn_epsilon_decay.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
