#!/usr/bin/env python3
"""
DQN Live Autoscaling Controller for 5G UPF.

Runs a control loop that:
  1. Trains DQN agent on historical CSV data (train_2-5.csv) at startup
  2. Queries Prometheus for current PPS every --interval seconds
  3. DQN agent picks action: scale_down / hold / scale_up
  4. Executes kubectl scale if replicas need to change
  5. Logs all decisions to CSV

Usage:
    # Train from CSVs and run live:
    python dqn_live_controller.py --train-dir manifests/autoscaling/dqn/training_data

    # Load pre-trained model (skip training):
    python dqn_live_controller.py --load-model manifests/autoscaling/dqn/dqn_model.pth

    # Dry run (no actual scaling):
    python dqn_live_controller.py --train-dir manifests/autoscaling/dqn/training_data --dry-run
"""

import argparse
import csv
import glob
import os
import random
import signal
import subprocess
import sys
import time
import warnings
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

#Constants
PROM_URL = os.environ.get(
    "PROM_URL",
    "http://kps-kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
)
DEPLOYMENT = "open5gs-upf1"
NAMESPACE  = "open5gs"

running = True

def signal_handler(sig, frame):
    global running
    print("\n[ctrl] Shutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


#DQN Model & Agent (self-contained, mirrors dqn_predictor.py)

class UPFScalingEnv:
    """Simulated UPF scaling environment for offline training."""
    def __init__(self, pps_series, threshold=4000, min_replicas=1, max_replicas=5):
        self.pps = pps_series
        self.threshold = threshold
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.n_steps = len(pps_series)
        self.reset()

    def reset(self):
        self.t = 0
        self.replicas = 1
        self.prev_replicas = 1
        return self._get_state()

    def _get_state(self):
        pps_now = self.pps[self.t] / self.threshold
        pps_trend = (self.pps[self.t] - self.pps[self.t - 1]) / self.threshold if self.t > 0 else 0.0
        replicas_norm = self.replicas / self.max_replicas
        return np.array([pps_now, pps_trend, replicas_norm], dtype=np.float32)

    def _ideal_replicas(self, pps):
        if pps <= 0:
            return 1
        return max(1, min(int(np.ceil(pps / self.threshold)), self.max_replicas))

    def step(self, action):
        self.prev_replicas = self.replicas
        if action == 0:
            self.replicas = max(self.min_replicas, self.replicas - 1)
        elif action == 2:
            self.replicas = min(self.max_replicas, self.replicas + 1)

        ideal = self._ideal_replicas(self.pps[self.t])
        reward = 0.0
        diff = abs(self.replicas - ideal)
        if diff == 0:
            reward += 10.0
        else:
            reward -= diff * 5.0
        if self.replicas != self.prev_replicas:
            reward += 2.0 if self.replicas == ideal else -3.0
        if self.replicas > ideal:
            reward -= (self.replicas - ideal) * 2.0
        if self.replicas < ideal:
            reward -= (ideal - self.replicas) * 3.0

        self.t += 1
        done = self.t >= self.n_steps
        next_state = self._get_state() if not done else np.zeros(3, dtype=np.float32)
        return next_state, reward, done


if HAS_TORCH:
    class DQN(nn.Module):
        def __init__(self, state_size=3, action_size=3, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, action_size),
            )
        def forward(self, x):
            return self.net(x)


class DQNAgent:
    def __init__(self, state_size=3, action_size=3, lr=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, memory_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        if HAS_TORCH:
            self.device = torch.device("cpu")
            self.policy_net = DQN(state_size, action_size).to(self.device)
            self.target_net = DQN(state_size, action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()
        else:
            self.q_table = {}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        if HAS_TORCH:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_t).argmax(dim=1).item()
        else:
            key = tuple(np.round(state, 2))
            if key not in self.q_table:
                return random.randrange(self.action_size)
            return int(np.argmax(self.q_table[key]))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        if HAS_TORCH:
            states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
            actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
            next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
            dones = torch.BoolTensor([b[4] for b in batch]).to(self.device)
            q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = self.target_net(next_states).max(dim=1)[0]
                q_next[dones] = 0.0
                q_target = rewards + self.gamma * q_next
            loss = self.loss_fn(q_current, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return 0.0

    def update_target(self):
        if HAS_TORCH:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        if HAS_TORCH:
            torch.save(self.policy_net.state_dict(), path)
            print(f"  [model] Saved weights to {path}")

    def load(self, path):
        if HAS_TORCH:
            self.policy_net.load_state_dict(torch.load(path, map_location="cpu"))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = 0.0  # pure exploitation
            print(f"  [model] Loaded weights from {path}")


#Training

def load_training_data(csv_path):
    """Load watcher CSV and extract PPS series."""
    df = pd.read_csv(csv_path)
    df["pps"] = pd.to_numeric(df["pps_prom"], errors="coerce").fillna(0)
    return df["pps"].values


def train_on_files(agent, csv_files, episodes_per_file, threshold, max_replicas):
    """Train DQN agent on multiple CSV files."""
    envs = []
    for f in csv_files:
        pps = load_training_data(f)
        if len(pps) < 5:
            print(f"  [warn] Skipping {os.path.basename(f)} (only {len(pps)} samples)")
            continue
        envs.append(UPFScalingEnv(pps, threshold=threshold, max_replicas=max_replicas))
        print(f"  Loaded: {os.path.basename(f)} ({len(pps)} samples, "
              f"PPS range: {pps.min():.0f}–{pps.max():.0f})")

    if not envs:
        print("  [error] No valid training files!")
        return []

    total_episodes = episodes_per_file * len(envs)
    print(f"\n  Training {total_episodes} episodes across {len(envs)} files...")

    rewards_history = []
    for ep in range(total_episodes):
        env = random.choice(envs)
        state = env.reset()
        total_reward = 0
        for t in range(env.n_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            if done:
                break
        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (ep + 1) % 10 == 0:
            agent.update_target()

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"    Episode {ep+1}/{total_episodes} | "
                  f"Avg reward (last 100): {avg:.1f} | Epsilon: {agent.epsilon:.3f}")

    print(f"\n  Training complete! Final Epsilon: {agent.epsilon:.4f}")
    print(f"  Avg reward (last 100): {np.mean(rewards_history[-100:]):.1f}")
    return rewards_history


# ── Kubernetes / Prometheus helpers (same as ARIMA live) ────────────

def query_prometheus_via_kubectl():
    """Query Prometheus from inside the cluster using kubectl run."""
    query = ('sum(rate(fivegs_ep_n3_gtp_indatapktn3upf'
             '{namespace="open5gs",service="open5gs-upf1-metrics"}[30s]))')
    pod_name = f"dqn-pq-{int(time.time()) % 100000}"
    try:
        cmd = [
            "kubectl", "run", "-n", "monitoring", "--rm", "-i",
            "--restart=Never", "--quiet", pod_name,
            "--image=curlimages/curl:latest", "--",
            "curl", "-fsS", f"{PROM_URL}/api/v1/query",
            "--data-urlencode", f"query={query}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            import json
            data = json.loads(result.stdout.strip())
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
    except Exception as e:
        print(f"  [warn] Prometheus query failed: {e}")
    return None


def get_pps_from_prometheus():
    """Get UPF PPS using kubectl-based Prometheus query."""
    pps = query_prometheus_via_kubectl()
    return pps if pps is not None else 0.0


def get_current_replicas():
    """Get current replica count from kubectl."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "deploy", DEPLOYMENT, "-n", NAMESPACE,
             "-o", "jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True, timeout=5
        )
        val = result.stdout.strip()
        return int(val) if val else 1
    except Exception:
        return 1


def scale_deployment(desired):
    """Execute kubectl scale."""
    try:
        result = subprocess.run(
            ["kubectl", "scale", "deploy", DEPLOYMENT, "-n", NAMESPACE,
             f"--replicas={desired}"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  [error] Scale failed: {e}")
        return False


#Main

ACTION_NAMES = {0: "scale down", 1: "hold", 2: "scale up"}


def main():
    parser = argparse.ArgumentParser(description="DQN Live Autoscaling Controller")
    parser.add_argument("--threshold", type=int, default=1500,
                        help="PPS per replica (default: 1,500)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Control loop interval in seconds (default: 5)")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Cooldown seconds between scale actions (default: 30)")
    parser.add_argument("--max-replicas", type=int, default=5)
    parser.add_argument("--train-dir", type=str,
                        default="manifests/autoscaling/dqn/training_data",
                        help="Directory with train_*.csv files")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Training episodes per file (default: 500)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to pre-trained model weights (.pth)")
    parser.add_argument("--save-model", type=str,
                        default="manifests/autoscaling/dqn/dqn_model.pth",
                        help="Path to save model weights after training")
    parser.add_argument("--log", type=str,
                        default="manifests/autoscaling/dqn/results/dqn_live_log.csv",
                        help="Output log CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually scale, just log decisions")
    args = parser.parse_args()

    # ── Initialize agent ──
    agent = DQNAgent(state_size=3, action_size=3, memory_size=10000)

    mode = "DRY-RUN" if args.dry_run else "LIVE"

    if args.load_model:
        # Skip training, load pre-trained weights
        print(f"""
===========================================================
DQN Live Autoscaling Controller

Mode:       {mode:<35s}
Model:      {os.path.basename(args.load_model):<35s}
Backend:    {'PyTorch' if HAS_TORCH else 'Q-Table':<35s}
Threshold:  {args.threshold:<35d}
Interval:   {args.interval}s{'':<33s}
Cooldown:   {args.cooldown}s{'':<33s}
Log:        {os.path.basename(args.log):<35s}
===========================================================
        """)
        agent.load(args.load_model)
    else:
        # Train from CSV files
        train_files = sorted(glob.glob(os.path.join(args.train_dir, "train_*.csv")))
        if not train_files:
            print(f"ERROR: No train_*.csv files in {args.train_dir}")
            sys.exit(1)

        print(f"""
============================================================
DQN Live Autoscaling Controller

Mode:       {mode:<35s}
Backend:    {'PyTorch' if HAS_TORCH else 'Q-Table':<35s}
Train files:{len(train_files):<35d}
Episodes:   {str(args.episodes) + " per file":<35s}
Threshold:  {args.threshold:<35d}
Interval:   {args.interval}s{'':<33s}
Cooldown:   {args.cooldown}s{'':<33s}
Log:        {os.path.basename(args.log):<35s}
============================================================
        """)

        print("[train] Loading training data...")
        rewards = train_on_files(agent, train_files, args.episodes,
                                 args.threshold, args.max_replicas)

        # Save model
        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
            agent.save(args.save_model)

        # Save training curve
        curve_path = args.log.replace(".csv", "_training_curve.csv")
        os.makedirs(os.path.dirname(curve_path) or ".", exist_ok=True)
        pd.DataFrame({"episode": range(len(rewards)),
                       "total_reward": rewards}).to_csv(curve_path, index=False)
        print(f"  [train] Training curve saved to {curve_path}")

    # Set epsilon to 0 for live mode (pure exploitation)
    agent.epsilon = 0.0
    print(f"\n[live] Agent epsilon set to 0 (greedy mode)")

    # ── Setup CSV log ──
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log_file = open(args.log, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "ts_iso", "pps_actual", "current_replicas",
        "desired_replicas", "dqn_action", "scale_executed"
    ])

    # ── Disable HPA ──
    if not args.dry_run:
        print("[init] Disabling HPA for UPF1...")
        subprocess.run(
            ["kubectl", "delete", "hpa", "open5gs-upf1-pps", "-n", NAMESPACE],
            capture_output=True, text=True
        )
        print("[init] HPA disabled. DQN is now in control.\n")
    else:
        print("[init] DRY-RUN mode — HPA not modified.\n")

    # ── Control loop ──
    step = 0
    prev_pps = 0.0
    last_scale_time = 0
    current_replicas = get_current_replicas()

    try:
        while running:
            loop_start = time.time()
            now = datetime.now(timezone.utc)
            ts = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")

            # 1. Get current PPS
            pps = get_pps_from_prometheus()
            current_replicas = get_current_replicas()

            # 2. Build DQN state
            pps_norm = pps / args.threshold
            pps_trend = (pps - prev_pps) / args.threshold if step > 0 else 0.0
            replicas_norm = current_replicas / args.max_replicas
            state = np.array([pps_norm, pps_trend, replicas_norm], dtype=np.float32)

            # 3. Agent picks action (epsilon=0, pure exploitation)
            action = agent.act(state)
            action_name = ACTION_NAMES[action]

            # 4. Compute desired replicas
            desired = current_replicas
            if action == 0:  # scale_down
                desired = max(1, current_replicas - 1)
            elif action == 2:  # scale_up
                desired = min(args.max_replicas, current_replicas + 1)

            # 5. Apply with cooldown
            scaled = False
            time_since_last = time.time() - last_scale_time

            if desired != current_replicas and time_since_last >= args.cooldown:
                if not args.dry_run:
                    scaled = scale_deployment(desired)
                    if scaled:
                        last_scale_time = time.time()
            elif desired == current_replicas:
                action_name = "hold"

            # 6. Log
            writer.writerow([ts, round(pps, 2), current_replicas,
                             desired, action_name, scaled])
            log_file.flush()

            # 7. Print status
            step += 1
            print(f"  [{step:>4d}] {ts} | PPS: {pps:>8.1f} | "
                  f"Replicas: {current_replicas} -> {desired} | "
                  f"{action_name:<10s}")

            prev_pps = pps
            elapsed = time.time() - loop_start
            time.sleep(max(0, args.interval - elapsed))

    finally:
        log_file.close()
        print(f"\n[done] Log saved to {args.log}")
        print(f"[done] {step} control loop iterations completed")

        if not args.dry_run:
            print("[cleanup] Re-enabling HPA...")
            subprocess.run(
                ["kubectl", "apply", "-f",
                 "manifests/autoscaling/hpa-upf1-pps.yaml"],
                capture_output=True, text=True
            )
            print("[cleanup] HPA re-enabled.")


if __name__ == "__main__":
    main()
