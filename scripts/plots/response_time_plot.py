#!/usr/bin/env python3
"""
Estimated Response Time using M/M/c Queuing Model
- X-axis: Time (seconds)
- Y-axis: Estimated Response Time (ms)
- Lines: one per pod count (1 to max_replicas used by algorithm)
  Each line shows: "if c pods were running, what would response time be?"
  → 1 pod = highest (overloaded during peak)
  → more pods = lower and smoother
- Separate plot for HPA, ARIMA, DQN
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import math
import os

# ── Config ────────────────────────────────────────────────────────────────────
MU        = 1500    # service rate per replica (PPS threshold)
MAX_RT_MS = 50.0    # cap for saturated/overloaded state (ms)
BASE_RT   = (1 / MU) * 1000  # ~0.667ms minimum service time

# ── Erlang-C ──────────────────────────────────────────────────────────────────
def erlang_c(c, lam, mu):
    rho = lam / (c * mu)
    if rho >= 1.0:
        return 1.0
    a = lam / mu
    num = (a ** c) / math.factorial(c) / (1 - rho)
    denom = sum((a ** k) / math.factorial(k) for k in range(c)) + num
    return num / denom

def response_time_ms(lam, c, mu=MU):
    """M/M/c mean response time in ms."""
    if lam <= 0:
        return BASE_RT
    rho = lam / (c * mu)
    if rho >= 1.0:
        return MAX_RT_MS   # system saturated
    ec = erlang_c(c, lam, mu)
    w_s = (1.0 / mu) + ec / (c * mu - lam)
    return min(w_s * 1000, MAX_RT_MS)

# ── Helper ────────────────────────────────────────────────────────────────────
def find_latest_folder(pattern, filename):
    folders = sorted(glob.glob(pattern), reverse=True)
    for f in folders:
        if os.path.isfile(f"{f}/{filename}"):
            return f
    raise FileNotFoundError(f"No folder matching '{pattern}' contains '{filename}'")

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_hpa():
    f = find_latest_folder("results/*-hpa-experiment", "watch.csv")
    print(f"  [HPA] folder: {f}")
    df = pd.read_csv(f"{f}/watch.csv", parse_dates=["ts_iso"])
    df = df.rename(columns={"pps_prom": "pps_actual", "hpa_replicas": "current_replicas"})
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True), f

def load_arima():
    f = find_latest_folder("results/*-arima-experiment", "arima_live.csv")
    print(f"  [ARIMA] folder: {f}")
    df = pd.read_csv(f"{f}/arima_live.csv", parse_dates=["ts_iso"])
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True), f

def load_dqn():
    folders = sorted(glob.glob("results/*-dqn-experiment"), reverse=True)
    for f in folders:
        if os.path.isfile(f"{f}/dqn_live.csv"):
            print(f"  [DQN] folder: {f}")
            df = pd.read_csv(f"{f}/dqn_live.csv", parse_dates=["ts_iso"])
            df["current_replicas"] = df["current_replicas"].clip(lower=1)
            return df.sort_values("ts_iso").reset_index(drop=True), f
    fallback = "manifests/autoscaling/dqn/results/dqn_live_experiment.csv"
    print(f"  [DQN] fallback: {fallback}")
    df = pd.read_csv(fallback, parse_dates=["ts_iso"])
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True), "manifests/autoscaling/dqn/results"

# ── Colors per pod count ──────────────────────────────────────────────────────
REPLICA_COLORS = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
#                  1 pod       2 pods     3 pods      4 pods     5 pods

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_response_time(df, title, out_path, pps_col="pps_actual"):
    t0 = df["ts_iso"].iloc[0]
    df = df.copy()
    df["elapsed"] = (df["ts_iso"] - t0).dt.total_seconds()

    max_replicas = int(df["current_replicas"].max())
    print(f"  [{title}] max replicas: {max_replicas}")

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Estimated Response Time — {title}", fontsize=14, fontweight="bold")

    # One line per pod count — shows what RT would be if c pods handled all traffic
    for c in range(1, max_replicas + 1):
        rt = df[pps_col].apply(lambda lam: response_time_ms(lam, c))
        ax.plot(df["elapsed"], rt,
                color=REPLICA_COLORS[c - 1],
                linewidth=2.0,
                label=f"{c} pod{'s' if c > 1 else ''}")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Estimated Response Time (ms)", fontsize=11)
    ax.set_ylim(0, MAX_RT_MS + 5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.legend(loc="upper right", fontsize=10, title="Number of Pods")
    ax.grid(True, alpha=0.3)

    # Annotate saturation threshold
    ax.axhline(y=MAX_RT_MS, color="red", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Saturated (overloaded)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating response time plots...")

    hpa_df,   hpa_dir   = load_hpa()
    arima_df, arima_dir = load_arima()
    dqn_df,   dqn_dir   = load_dqn()

    plot_response_time(hpa_df,   "HPA",   f"{hpa_dir}/hpa_response_time.png")
    plot_response_time(arima_df, "ARIMA", f"{arima_dir}/arima_response_time.png")
    plot_response_time(dqn_df,   "DQN",   f"{dqn_dir}/dqn_response_time.png")

    print("\nDone! All 3 response time graphs generated.")
