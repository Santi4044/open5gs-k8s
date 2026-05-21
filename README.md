# 5G Core Network Autoscaling — Algorithm Performance Comparison

> **Goal:** Compare three autoscaling strategies — **HPA**, **ARIMA**, and **DQN** — for the User Plane Function (UPF) of a 5G Core Network deployed on Kubernetes using [Open5GS](https://open5gs.org/) and [UERANSIM](https://github.com/aligungr/UERANSIM).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment & Prerequisites](#2-environment--prerequisites)
3. [Repository Structure](#3-repository-structure)
4. [Algorithms Overview](#4-algorithms-overview)
5. [Running the Experiments](#5-running-the-experiments)
   - [HPA](#51-hpa-horizontal-pod-autoscaler)
   - [ARIMA](#52-arima)
   - [DQN](#53-dqn-deep-q-network)
6. [Configuration Reference](#6-configuration-reference)
   - [HPA Config](#61-hpa)
   - [ARIMA Config](#62-arima)
   - [DQN Config](#63-dqn)
7. [Results](#7-results)

---

## 1. Project Overview

This project evaluates and compares three autoscaling approaches applied to the **UPF (User Plane Function)** of a 5G Core Network:

| Algorithm | Type | Description |
|-----------|------|-------------|
| **HPA** | Reactive | Kubernetes-native; scales based on a live PPS threshold |
| **ARIMA** | Predictive | Time-series forecasting; predicts future PPS to scale proactively |
| **DQN** | Reinforcement Learning | Learns an optimal scaling policy from historical traffic patterns |

Each algorithm drives pod scaling for the `open5gs-upf1` deployment in the `open5gs` Kubernetes namespace. Traffic is generated via `iperf3` through UERANSIM UE pods, and metrics are scraped from **Prometheus**. All experiments follow the same traffic pattern:

```
IDLE (30s) → LOW traffic 10 Mbps (60s) → IDLE (30s) → HIGH traffic 40 Mbps (120s) → IDLE (120s)
```

---

## 2. Environment & Prerequisites

> This section is intentionally brief — the focus of this repo is the autoscaling comparison, not the 5G setup itself.

The experiments assume a working environment with:

- **Kubernetes cluster** with the `open5gs` namespace deployed (Open5GS 5G Core + UERANSIM)
- **Prometheus** running in the `monitoring` namespace (e.g., via `kube-prometheus-stack`)
- **Custom metrics adapter** configured so the `upf1_n3_in_pps` metric is accessible to Kubernetes HPA
- **Python 3.x** installed locally with dependencies from `requirements.txt`
- **kubectl** configured and pointing to your cluster

To install Python dependencies for ARIMA/DQN:

```bash
pip install -r manifests/autoscaling/arima/requirements.txt
```

> For full 5G infrastructure setup, refer to the original deployment scripts (`deploy-core.sh`, `deploy-ran.sh`) and the Kubernetes manifests under `open5gs/` and `ueransim/`.

---

## 3. Repository Structure

```
open5gs-k8s/
│
├── manifests/
│   ├── autoscaling/
│   │   ├── hpa-upf1-pps.yaml          # HPA manifest for UPF1
│   │   ├── hpa-upf2-pps.yaml          # HPA manifest for UPF2
│   │   ├── arima/
│   │   │   ├── arima_live_controller.py   # ARIMA control loop
│   │   │   └── requirements.txt           # Python deps for ARIMA
│   │   └── dqn/
│   │       ├── dqn_live_controller.py     # DQN control loop (train + live)
│   │       ├── dqn_model.pth              # Pre-trained DQN model weights
│   │       └── training_data/             # CSV files used for offline training
│   ├── monitoring/                    # Prometheus / Grafana manifests
│   └── custom-metrics/                # Custom metrics adapter config
│
├── scripts/
│   ├── traffic/
│   │   ├── run_hpa_experiment.sh          # Run full HPA experiment
│   │   ├── run_arima_live_experiment.sh   # Run full ARIMA experiment
│   │   ├── run_dqn_live_experiment.sh     # Run full DQN experiment
│   │   ├── run_dqn_training.sh            # Offline DQN training
│   │   ├── run_collect_training_data.sh   # Collect DQN training data
│   │   ├── run_iperf_udp.sh               # Single iperf3 UDP burst
│   │   ├── run_pattern{1,2,3}.sh          # Traffic patterns for training
│   │   └── watch_scaling_prom.sh          # Prometheus metric watcher (CSV)
│   └── plots/                         # Plot generation scripts
│
├── results/                           # Experiment output (auto-created per run)
│   ├── <timestamp>-hpa-experiment/
│   ├── <timestamp>-arima-experiment/
│   ├── <timestamp>-dqn-experiment/
│   ├── <timestamp>-dqn-training/
│   └── *.png                          # Comparison plots
│
├── open5gs/                           # Open5GS K8s manifests
├── ueransim/                          # UERANSIM K8s manifests
├── deploy-core.sh                     # Deploy 5G core
├── deploy-ran.sh                      # Deploy RAN (UERANSIM)
└── requirements.txt                   # Python deps for utility scripts
```

---

## 4. Algorithms Overview

### 4.1 HPA — Horizontal Pod Autoscaler

Kubernetes' built-in autoscaler. It watches the **`upf1_n3_in_pps`** custom metric (packets per second ingested by UPF1's N3 interface, served via a custom metrics adapter from Prometheus) and scales the deployment reactively when the metric crosses the configured threshold.

- **Pros:** Zero extra components, native K8s, no training required
- **Cons:** Purely reactive — always one step behind traffic spikes

### 4.2 ARIMA — AutoRegressive Integrated Moving Average

A Python control loop (`arima_live_controller.py`) that:
1. Queries Prometheus every `--interval` seconds for the current PPS
2. Maintains a **sliding window** of PPS history
3. Fits an **ARIMA(p,d,q)** model on the window and forecasts `--horizon` steps ahead
4. Computes desired replicas from the forecasted PPS value
5. Calls `kubectl scale` if a change is needed (respecting a cooldown)
6. Disables HPA while running, and re-enables it on exit

- **Pros:** Proactive — acts before the spike hits
- **Cons:** Sensitive to non-stationary or abrupt traffic changes; ARIMA order needs tuning

### 4.3 DQN — Deep Q-Network

A reinforcement learning controller (`dqn_live_controller.py`) that:
1. Learns a scaling **policy** from historical traffic data (offline training)
2. At runtime, observes the current PPS + replica state and selects an action (scale up / hold / scale down)
3. Uses a neural network (Q-network) to estimate action values
4. Ships with a **pre-trained model** (`dqn_model.pth`) so you can skip training

- **Pros:** Can generalise across complex traffic patterns; improves with more data
- **Cons:** Requires training data collection; less interpretable than ARIMA or HPA

---

## 5. Running the Experiments

All experiment scripts are run from the **repo root**. Each one follows the same traffic phases and saves timestamped output to `results/`.

### 5.1 HPA — Horizontal Pod Autoscaler

**Step 1:** Apply the HPA manifest.

```bash
kubectl apply -f manifests/autoscaling/hpa-upf1-pps.yaml
```

**Step 2:** Run the experiment.

```bash
bash scripts/traffic/run_hpa_experiment.sh
```

This script:
- Starts a Prometheus metric watcher in the background (saves `watch.csv`)
- Runs the traffic phases via `iperf3` through the UE pod
- Saves results to `results/<timestamp>-hpa-experiment/`

**Optional — override parameters via environment variables:**

```bash
LOW_RATE=20M PEAK_RATE=60M PEAK_DUR=180 bash scripts/traffic/run_hpa_experiment.sh
```

---

### 5.2 ARIMA

> Make sure the HPA is **not** applied before running this. The ARIMA controller disables it automatically, but it's cleaner to start without it.

```bash
kubectl delete hpa open5gs-upf1-pps -n open5gs --ignore-not-found
```

**Run the experiment:**

```bash
bash scripts/traffic/run_arima_live_experiment.sh
```

This script:
- Starts `arima_live_controller.py` with default parameters in the background
- Runs the same traffic phases as the HPA experiment
- Saves `arima_live.csv` (per-step decisions) and `traffic.log` to `results/<timestamp>-arima-experiment/`
- Stops the controller and **re-applies the HPA** on exit

**Run the controller manually with custom parameters:**

```bash
python manifests/autoscaling/arima/arima_live_controller.py \
  --interval 5 \
  --threshold 1500 \
  --horizon 3 \
  --order 2 1 2 \
  --window 30 \
  --min-window 10 \
  --cooldown 30 \
  --log results/my-arima-run/arima_live.csv
```

---

### 5.3 DQN — Deep Q-Network

The DQN workflow has three stages: **collect training data → train → run live experiment**. A pre-trained model (`dqn_model.pth`) is included so you can skip straight to Step 3 if desired.

**Step 1 (optional): Collect training data**

```bash
bash scripts/traffic/run_collect_training_data.sh
```

Runs three traffic patterns and saves CSVs to `manifests/autoscaling/dqn/training_data/`.

**Step 2 (optional): Train the DQN model**

```bash
bash scripts/traffic/run_dqn_training.sh
```

Trains from the collected CSVs and saves the model to `manifests/autoscaling/dqn/dqn_model.pth`, overwriting the existing one.

**Step 3: Run the live experiment**

```bash
bash scripts/traffic/run_dqn_live_experiment.sh
```

This script:
- Starts a local `iperf3` server
- Loads the pre-trained model and starts `dqn_live_controller.py` in the background
- Runs the same traffic phases (IDLE → LOW → IDLE → HIGH → IDLE)
- Saves `dqn_live.csv` to `results/<timestamp>-dqn-experiment/`

**Run the controller manually:**

```bash
python manifests/autoscaling/dqn/dqn_live_controller.py \
  --interval 5 \
  --threshold 1500 \
  --cooldown 30 \
  --load-model manifests/autoscaling/dqn/dqn_model.pth \
  --log results/my-dqn-run/dqn_live.csv
```

---

## 6. Configuration Reference

### 6.1 HPA

**File:** `manifests/autoscaling/hpa-upf1-pps.yaml`

| Field | Default | Description |
|-------|---------|-------------|
| `spec.minReplicas` | `1` | Minimum UPF1 replicas |
| `spec.maxReplicas` | `5` | Maximum UPF1 replicas |
| `metrics[].external.target.value` | `"1500"` | PPS threshold per replica — scale up when exceeded |
| `behavior.scaleUp.stabilizationWindowSeconds` | `0` | No delay on scale-up |
| `behavior.scaleDown.stabilizationWindowSeconds` | `0` | No delay on scale-down |
| `behavior.*.policies[].periodSeconds` | `30` | Cooldown period between scaling events |

Edit the file and re-apply with `kubectl apply -f manifests/autoscaling/hpa-upf1-pps.yaml`.

---

### 6.2 ARIMA

**File:** `manifests/autoscaling/arima/arima_live_controller.py`

All parameters are exposed as CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--threshold` | `1500` | PPS per replica — used to convert forecasted PPS to replica count |
| `--interval` | `5` | Control loop frequency (seconds) |
| `--horizon` | `3` | ARIMA forecast horizon (number of steps ahead) |
| `--order P D Q` | `2 1 2` | ARIMA(p,d,q) order |
| `--window` | `30` | Sliding window size (number of samples) |
| `--min-window` | `10` | Minimum samples before ARIMA activates (uses current PPS before this) |
| `--max-replicas` | `5` | Maximum replicas the controller will request |
| `--cooldown` | `30` | Seconds to wait between consecutive scale actions |
| `--log` | *(path)* | CSV file path for decision log |
| `--dry-run` | off | Log decisions without actually scaling |

To tune the ARIMA model, adjust `--order`, `--window`, and `--horizon`. Larger windows give a smoother fit; a higher horizon makes the controller more proactive but also more sensitive to forecast error.

---

### 6.3 DQN

**File:** `manifests/autoscaling/dqn/dqn_live_controller.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--threshold` | `1500` | PPS per replica (same meaning as ARIMA/HPA) |
| `--interval` | `5` | Control loop frequency (seconds) |
| `--cooldown` | `30` | Seconds between scale actions |
| `--load-model` | *(path)* | Path to pre-trained `.pth` model weights |
| `--save-model` | *(path)* | Where to save a newly trained model |
| `--train-dir` | *(path)* | Directory of training CSVs (used with `--train-only`) |
| `--train-only` | off | Run offline training then exit (no live control loop) |
| `--log` | *(path)* | CSV file path for decision log |

**Pre-trained model:** `manifests/autoscaling/dqn/dqn_model.pth`

**Training data:** `manifests/autoscaling/dqn/training_data/` — CSV files produced by `run_collect_training_data.sh`.

---

## 7. Results

Each experiment run creates a timestamped folder under `results/`:

```
results/
├── <timestamp>-hpa-experiment/
│   ├── watch.csv        # time-series: PPS, replicas
│   └── traffic.log      # iperf3 output per phase
├── <timestamp>-arima-experiment/
│   ├── arima_live.csv   # ts, pps_actual, pps_forecast, replicas, action
│   └── traffic.log
└── <timestamp>-dqn-experiment/
    └── dqn_live.csv     # ts, pps, replicas, action, reward
```

Pre-generated comparison plots are stored directly in `results/`:

| Plot | Description |
|------|-------------|
| `results/combined_response_time.png` | Response time over time for all three algorithms |
| `results/rt_vs_request_rate.png` | Response time vs. request rate comparison |

To regenerate plots, use the scripts in `scripts/plots/`.

---

## License

This project builds on [open5gs-k8s](https://github.com/niloysh/open5gs-k8s) by Niloy Saha. See [LICENSE](LICENSE) for details.
