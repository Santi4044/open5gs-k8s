#!/usr/bin/env python3
"""
Combined Estimated Response Time — Autoscaling Comparison
- X-axis: Time (seconds)
- Y-axis: Estimated Response Time (ms)
- Lines: No Autoscaling (1 pod), HPA, ARIMA, DQN
- Phase annotations using HPA timestamps as reference
- Output: results/combined_response_time.png
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

# ── Phase definitions (HPA as reference) ─────────────────────────────────────
HPA_PHASES = [
    (0,   "IDLE\n(30s)"),
    (30,  "LOW\n(10M/60s)"),
    (90,  "IDLE\n(30s)"),
    (120, "HIGH\n(40M/120s)"),
    (240, "IDLE\n(120s)"),
]

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_NO_AUTOSCALING = "#F44336"   # red
COLOR_HPA            = "#2196F3"   # blue
COLOR_ARIMA          = "#FF9800"   # orange
COLOR_DQN            = "#4CAF50"   # green

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
        return MAX_RT_MS
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
    return df.sort_values("ts_iso").reset_index(drop=True)

def load_arima():
    f = find_latest_folder("results/*-arima-experiment", "arima_live.csv")
    print(f"  [ARIMA] folder: {f}")
    df = pd.read_csv(f"{f}/arima_live.csv", parse_dates=["ts_iso"])
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True)

def load_dqn():
    folders = sorted(glob.glob("results/*-dqn-experiment"), reverse=True)
    for f in folders:
        if os.path.isfile(f"{f}/dqn_live.csv"):
            print(f"  [DQN] folder: {f}")
            df = pd.read_csv(f"{f}/dqn_live.csv", parse_dates=["ts_iso"])
            df["current_replicas"] = df["current_replicas"].clip(lower=1)
            return df.sort_values("ts_iso").reset_index(drop=True)
    fallback = "manifests/autoscaling/dqn/results/dqn_live_experiment.csv"
    print(f"  [DQN] fallback: {fallback}")
    df = pd.read_csv(fallback, parse_dates=["ts_iso"])
    df["current_replicas"] = df["current_replicas"].clip(lower=1)
    return df.sort_values("ts_iso").reset_index(drop=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_combined(hpa_df, arima_df, dqn_df):
    # Normalize elapsed time for each dataset independently
    for df in (hpa_df, arima_df, dqn_df):
        df["elapsed"] = (df["ts_iso"] - df["ts_iso"].iloc[0]).dt.total_seconds()

    # Compute response times
    def compute_rt(df):
        return [response_time_ms(lam, c)
                for lam, c in zip(df["pps_actual"], df["current_replicas"])]

    hpa_df["rt"]   = compute_rt(hpa_df)
    arima_df["rt"] = compute_rt(arima_df)
    dqn_df["rt"]   = compute_rt(dqn_df)

    # No Autoscaling: fixed c=1, use HPA's PPS as reference traffic
    hpa_df["rt_no_autoscaling"] = hpa_df["pps_actual"].apply(
        lambda lam: response_time_ms(lam, 1)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Estimated Response Time — Autoscaling Comparison",
                 fontsize=14, fontweight="bold")

    ax.plot(hpa_df["elapsed"], hpa_df["rt_no_autoscaling"],
            color=COLOR_NO_AUTOSCALING, linewidth=2, linestyle="--",
            label="No Autoscaling (1 pod)")
    ax.plot(hpa_df["elapsed"], hpa_df["rt"],
            color=COLOR_HPA, linewidth=2, linestyle="-",
            label="HPA")
    ax.plot(arima_df["elapsed"], arima_df["rt"],
            color=COLOR_ARIMA, linewidth=2, linestyle="-",
            label="ARIMA")
    ax.plot(dqn_df["elapsed"], dqn_df["rt"],
            color=COLOR_DQN, linewidth=2, linestyle="-",
            label="DQN")

    # ── Phase annotations (HPA timestamps as reference) ───────────────────────
    for x, label in HPA_PHASES:
        ax.axvline(x, color="gray", linewidth=0.8, linestyle="--")
        ax.text(x + 2, (MAX_RT_MS + 5) * 0.92, label,
                fontsize=7.5, color="gray", va="top")

    # Saturation reference line
    ax.axhline(y=MAX_RT_MS, color="red", linewidth=0.8, linestyle=":",
               alpha=0.4, label=f"Saturation ({MAX_RT_MS:.0f} ms)")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Estimated Response Time (ms)", fontsize=11)
    ax.set_ylim(0, MAX_RT_MS + 5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs("results", exist_ok=True)
    out_path = "results/combined_response_time.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating combined response time plot...")

    hpa_df   = load_hpa()
    arima_df = load_arima()
    dqn_df   = load_dqn()

    plot_combined(hpa_df, arima_df, dqn_df)
