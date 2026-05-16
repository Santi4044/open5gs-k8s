#!/usr/bin/env python3
"""
Estimated Response Time vs Request Rate (PPS)
- X-axis: Request Rate (PPS)
- Y-axis: Estimated Response Time (ms)
- Lines: No Autoscaling, HPA, ARIMA, DQN
- Based on M/M/c queuing model with integer replica step functions matching each algorithm's real scaling behaviour from experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import os

# Config
MU        = 1500
MAX_RT_MS = 50.0
BASE_RT   = (1 / MU) * 1000

# Erlang-C formula to calculate queueing delay
def erlang_c(c, lam, mu):
    rho = lam / (c * mu)
    if rho >= 1.0:
        return 1.0
    a = lam / mu
    num = (a ** c) / math.factorial(c) / (1 - rho)
    denom = sum((a ** k) / math.factorial(k) for k in range(c)) + num
    return num / denom

def response_time_ms(lam, c, mu=MU):
    c = max(1, int(round(c)))
    if lam <= 0:
        return BASE_RT
    rho = lam / (c * mu)
    if rho >= 1.0:
        return MAX_RT_MS
    ec = erlang_c(c, lam, mu)
    w_s = (1.0 / mu) + ec / (c * mu - lam)
    return min(w_s * 1000, MAX_RT_MS)

# Replica step functions
# Based on real experiment results:
# HPA - reactive delay, over-provisioned to 5 pods
# ARIMA - predicts early (~300 PPS before threshold), max 4 pods
# DQN - scales precisely at threshold, max 3 pods (no overshoot)

def replicas_no_autoscale(pps): return 1

def replicas_hpa(pps):
    if pps < 1800:   return 1
    elif pps < 3300: return 2
    elif pps < 4300: return 3
    elif pps < 5500: return 4
    else:            return 5

def replicas_arima(pps):
    if pps < 1200:   return 1
    elif pps < 2700: return 2
    elif pps < 4000: return 3
    else:            return 4

def replicas_dqn(pps):
    if pps < 1500:   return 1
    elif pps < 3000: return 2
    else:            return 3

# Generate request rate values
pps_range = np.linspace(0, 4400, 4000)

rt_no_autoscale = np.array([response_time_ms(p, replicas_no_autoscale(p)) for p in pps_range])
rt_hpa          = np.array([response_time_ms(p, replicas_hpa(p))          for p in pps_range])
rt_arima        = np.array([response_time_ms(p, replicas_arima(p))        for p in pps_range])
rt_dqn          = np.array([response_time_ms(p, replicas_dqn(p))          for p in pps_range])

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Estimated Response Time vs Request Rate",
             fontsize=14, fontweight="bold")

ax.plot(pps_range, rt_no_autoscale,
        color="#F44336", linewidth=2.5, linestyle="--",
        label="No Autoscaling (1 pod)")
ax.plot(pps_range, rt_hpa,
        color="#2196F3", linewidth=2.5, linestyle="-",
        label="HPA")
ax.plot(pps_range, rt_arima,
        color="#FF9800", linewidth=2.5, linestyle="-",
        label="ARIMA")
ax.plot(pps_range, rt_dqn,
        color="#4CAF50", linewidth=2.5, linestyle="-",
        label="DQN")

ax.axvline(x=1500, color="gray", linewidth=1.0, linestyle=":",
           label="Scaling threshold (1,500 PPS)")
ax.axvspan(1500, 4000, alpha=0.04, color="gray")
ax.text(2600, MAX_RT_MS * 0.92, "HIGH traffic region",
        fontsize=8, color="gray", ha="center", va="top")

ax.set_xlabel("Request Rate (PPS)", fontsize=11)
ax.set_ylabel("Estimated Response Time (ms)", fontsize=11)
ax.set_ylim(0, MAX_RT_MS + 5)
ax.set_xlim(0, 4400)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=MAX_RT_MS, color="#F44336", linewidth=0.7, linestyle=":", alpha=0.4)

os.makedirs("results", exist_ok=True)
out = "results/rt_vs_request_rate.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
