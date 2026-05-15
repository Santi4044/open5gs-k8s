#!/usr/bin/env python3
"""
Combined Estimated Response Time — Autoscaling Comparison
Plots HPA, ARIMA, DQN actual response time vs No Autoscaling baseline (1 pod fixed)
using M/M/c queuing model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import math
import os

# ── Config ────────────────────────────────────────────────────────────────────
MU        = 1500
MAX_RT_MS = 50.0
BASE_RT   = (1 / MU) * 1000

# ── Phase lines (HPA as reference) ───────────────────────────────────────────
PHASES = [
    (0,   "IDLE\n(30s)"),
    (30,  "LOW\n(10M/60s)"),
    (90,  "IDLE\n(30s)"),
    (120, "HIGH\n(40M/120s)"),
    (240, "IDLE\n(120s)"),
]

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
    if lam <= 0:
        return BASE_RT
    rho = lam / (c * mu)
    if rho >= 1.0:
        return MAX_RT_MS
    ec = erlang_c(c, lam, mu)
    w_s = (1.0 / mu) + ec / (c * mu - lam)
    return min(w_s * 1000, MAX_RT_MS)

# ── Loaders ───────────────────────────────────────────────────────────────────
def find_latest_folder(pattern, filename):
    folders = sorted(glob.glob(pattern), reverse=True)
    for f in folders:
        if os.path.isfile(f"{f}/{filename}"):
            return f
    raise FileNotFoundError(f"No folder matching '{pattern}' contains '{filename}'")

def load_hpa():
    f = find_latest_folder("results/*-hpa-experiment", "watch.csv")
    print(f"  [HPA] {f}")
    df = pd.read_csv(f"{f}/watch.csv", parse_dates=["ts_iso"])
    df = df.rename(columns={"pps_prom": "pps_actual", "hpa_replicas": "current_replicas"})
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True)

def load_arima():
    f = find_latest_folder("results/*-arima-experiment", "arima_live.csv")
    print(f"  [ARIMA] {f}")
    df = pd.read_csv(f"{f}/arima_live.csv", parse_dates=["ts_iso"])
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True)

def load_dqn():
    folders = sorted(glob.glob("results/*-dqn-experiment"), reverse=True)
    for f in folders:
        if os.path.isfile(f"{f}/dqn_live.csv"):
            print(f"  [DQN] {f}")
            df = pd.read_csv(f"{f}/dqn_live.csv", parse_dates=["ts_iso"])
            df["current_replicas"] = df["current_replicas"].clip(lower=1)
            return df.sort_values("ts_iso").reset_index(drop=True)
    fallback = "manifests/autoscaling/dqn/results/dqn_live_experiment.csv"
    print(f"  [DQN] fallback: {fallback}")
    df = pd.read_csv(fallback, parse_dates=["ts_iso"])
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True)

# ── Compute elapsed + response time ──────────────────────────────────────────
def add_elapsed_and_rt(df, pps_col="pps_actual"):
    df = df.copy()
    df["elapsed"] = (df["ts_iso"] - df["ts_iso"].iloc[0]).dt.total_seconds()
    df["rt"] = df.apply(lambda row: response_time_ms(row[pps_col], int(row["current_replicas"])), axis=1)
    df["rt_no_autoscale"] = df[pps_col].apply(lambda lam: response_time_ms(lam, 1))
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    hpa_df   = add_elapsed_and_rt(load_hpa())
    arima_df = add_elapsed_and_rt(load_arima())
    dqn_df   = add_elapsed_and_rt(load_dqn())

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Estimated Response Time — Autoscaling Comparison",
                 fontsize=14, fontweight="bold")

    # No autoscaling baseline (from HPA data, fixed 1 pod)
    ax.plot(hpa_df["elapsed"], hpa_df["rt_no_autoscale"],
            color="#F44336", linewidth=2, linestyle="--", label="No Autoscaling (1 pod)")

    # Each algorithm's actual response time
    ax.plot(hpa_df["elapsed"],   hpa_df["rt"],
            color="#2196F3", linewidth=2, linestyle="-", label="HPA")
    ax.plot(arima_df["elapsed"], arima_df["rt"],
            color="#FF9800", linewidth=2, linestyle="-", label="ARIMA")
    ax.plot(dqn_df["elapsed"],   dqn_df["rt"],
            color="#4CAF50", linewidth=2, linestyle="-", label="DQN")

    # Phase annotations
    for x, label in PHASES:
        ax.axvline(x, color="gray", linewidth=0.8, linestyle=":")
        ax.text(x + 2, (MAX_RT_MS + 5) * 0.92, label,
                fontsize=7.5, color="gray", va="top")

    # Saturation reference line
    ax.axhline(y=MAX_RT_MS, color="red", linewidth=0.8, linestyle=":", alpha=0.4)

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Estimated Response Time (ms)", fontsize=11)
    ax.set_ylim(0, MAX_RT_MS + 5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/combined_response_time.png", dpi=150, bbox_inches="tight")
    print("Saved: results/combined_response_time.png")
