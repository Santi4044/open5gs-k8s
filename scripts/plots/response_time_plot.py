#!/usr/bin/env python3
"""
Estimated Response Time using M/M/c Queuing Model
- X-axis: Time (seconds)
- Y-axis: Estimated Response Time (ms)
- Lines: 1 to max_replicas used (from actual data) + actual algorithm response time
- Separate plot for HPA, ARIMA, DQN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import math
import os

# ── Config ────────────────────────────────────────────────────────────────────
MU          = 1500        # service rate per replica (PPS)
BASE_MS     = 5.0         # base service time in ms
MAX_RT_MS   = 200.0       # cap response time for readability

# ── Erlang-C formula ──────────────────────────────────────────────────────────
def erlang_c(c, lam, mu):
    rho = lam / (c * mu)
    if rho >= 1.0:
        return 1.0
    a = lam / mu
    num = (a ** c) / math.factorial(c) * (1 / (1 - rho))
    denom = sum((a ** k) / math.factorial(k) for k in range(c)) + num
    return num / denom

def response_time_ms(lam, c, mu=MU, base_ms=BASE_MS):
    if lam <= 0:
        return base_ms
    rho = lam / (c * mu)
    if rho >= 1.0:
        return MAX_RT_MS
    ec = erlang_c(c, lam, mu)
    w_seconds = (1 / mu) + ec / (c * mu - lam)
    w_ms = w_seconds * 1000 * base_ms * mu
    return min(w_ms, MAX_RT_MS)

# ── Load datasets ─────────────────────────────────────────────────────────────
def load_hpa():
    folders = sorted(glob.glob("results/*-hpa-experiment"))
    if not folders:
        raise FileNotFoundError("No HPA results found in results/")
    f = folders[-1]
    df = pd.read_csv(f"{f}/watch.csv", parse_dates=["ts_iso"])
    df = df.rename(columns={"pps_prom": "pps_actual", "hpa_replicas": "current_replicas"})
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True), f

def load_arima():
    folders = sorted(glob.glob("results/*-arima-experiment"))
    if not folders:
        raise FileNotFoundError("No ARIMA results found in results/")
    f = folders[-1]
    df = pd.read_csv(f"{f}/arima_live.csv", parse_dates=["ts_iso"])
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True), f

def load_dqn():
    folders = sorted(glob.glob("results/*-dqn-experiment"))
    if not folders:
        df = pd.read_csv("manifests/autoscaling/dqn/results/dqn_live_experiment.csv",
                         parse_dates=["ts_iso"])
        df["current_replicas"] = df["current_replicas"].clip(lower=1)
        return df.sort_values("ts_iso").reset_index(drop=True), "manifests/autoscaling/dqn/results"
    f = folders[-1]
    df = pd.read_csv(f"{f}/dqn_live.csv", parse_dates=["ts_iso"])
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True), f

# ── Color palette — only as many as needed ───────────────────────────────────
REPLICA_COLORS = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
#                  1 pod       2 pods     3 pods      4 pods     5 pods

# ── Plot function ─────────────────────────────────────────────────────────────
def plot_response_time(df, title, out_path, pps_col="pps_actual"):
    t0 = df["ts_iso"].iloc[0]
    df["elapsed"] = (df["ts_iso"] - t0).dt.total_seconds()

    # ── Derive max replicas from actual data ──────────────────────────────────
    max_replicas = int(df["current_replicas"].max())
    print(f"  [{title}] max replicas used: {max_replicas}")

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Estimated Response Time — {title}", fontsize=14, fontweight="bold")

    # ── Theoretical lines for 1 to max_replicas only ─────────────────────────
    for c in range(1, max_replicas + 1):
        rt = df[pps_col].apply(lambda lam: response_time_ms(lam, c))
        ax.plot(df["elapsed"], rt,
                color=REPLICA_COLORS[c - 1],
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
                label=f"{c} pod{'s' if c > 1 else ''} (theoretical)")

    # ── Actual algorithm response time (black solid line) ─────────────────────
    actual_rt = df.apply(
        lambda row: response_time_ms(row[pps_col], int(row["current_replicas"])), axis=1)
    ax.plot(df["elapsed"], actual_rt,
            color="black",
            linewidth=2.5,
            linestyle="-",
            label=f"{title} actual")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Estimated Response Time (ms)", fontsize=11)
    ax.set_ylim(0, MAX_RT_MS + 10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")

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
