#!/usr/bin/env python3
"""
DQN-based Autoscaling Agent for 5G UPF.

Supports two modes:
  1. Single file:  python dqn_predictor.py data.csv
  2. Train/Test:   python dqn_predictor.py --train-dir DIR --test FILE

State:  [current_pps, pps_trend, current_replicas]
Action: 0=scale_down, 1=hold, 2=scale_up
"""

import argparse
import glob
import os
import random
import sys
import warnings
from collections import deque

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


class UPFScalingEnv:
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


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts_iso"])
    df = df.set_index("ts").sort_index()
    df["pps"] = pd.to_numeric(df["pps_prom"], errors="coerce").fillna(0)
    df["replicas"] = pd.to_numeric(df["hpa_replicas"], errors="coerce").fillna(1).astype(int)
    return df


def train_on_files(agent, csv_files, episodes_per_file, threshold):
    """Train agent across multiple CSV files (avoids overfitting)."""
    envs = []
    for f in csv_files:
        df = load_data(f)
        envs.append(UPFScalingEnv(df["pps"].values, threshold=threshold))
        print(f"  Loaded training file: {os.path.basename(f)} ({len(df)} samples)")

    total_episodes = episodes_per_file * len(csv_files)
    print(f"\n  Training {total_episodes} episodes across {len(envs)} files...")

    rewards_history = []
    for ep in range(total_episodes):
        # Randomly pick a training environment each episode
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
        if ep % 10 == 0:
            agent.update_target()
        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(f"    Episode {ep+1}/{total_episodes} | "
                  f"Avg Reward: {avg_r:.1f} | Epsilon: {agent.epsilon:.3f}")

    return rewards_history


def evaluate_agent(agent, env, df):
    state = env.reset()
    agent.epsilon = 0.0
    results = []
    for t in range(env.n_steps):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        ideal = env._ideal_replicas(env.pps[t])
        action_names = ["scale_down", "hold", "scale_up"]
        results.append({
            "ts": df.index[t],
            "actual_pps": env.pps[t],
            "actual_replicas": df["replicas"].iloc[t],
            "dqn_replicas": env.replicas,
            "dqn_action": action_names[action],
            "ideal_replicas": ideal,
            "replica_match_hpa": env.replicas == df["replicas"].iloc[t],
            "replica_match_ideal": env.replicas == ideal,
            "reward": round(reward, 2),
        })
        state = next_state
        if done:
            break
    return pd.DataFrame(results)


def compute_metrics(results_df):
    v = results_df.dropna()
    return {
        "total_samples": len(v),
        "match_vs_hpa_pct": round(v["replica_match_hpa"].mean() * 100, 1),
        "match_vs_ideal_pct": round(v["replica_match_ideal"].mean() * 100, 1),
        "total_reward": round(v["reward"].sum(), 2),
        "avg_reward": round(v["reward"].mean(), 2),
        "dqn_scale_events": int((v["dqn_replicas"].diff().fillna(0) != 0).sum()),
        "hpa_scale_events": int((v["actual_replicas"].diff().fillna(0) != 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="DQN Autoscaling Agent")
    parser.add_argument("csv_file", nargs="?", default=None,
                        help="Single CSV (train+test on same file)")
    parser.add_argument("--train-dir", default=None,
                        help="Directory with train_*.csv files")
    parser.add_argument("--test", default=None,
                        help="Test CSV file (unseen data)")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Episodes per training file (default: 500)")
    parser.add_argument("--threshold", type=int, default=4000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"DQN Autoscaling Agent")
    print(f"{'='*50}")
    print(f"Backend: {'PyTorch' if HAS_TORCH else 'Q-Table (fallback)'}")

    agent = DQNAgent(state_size=3, action_size=3, memory_size=10000)

    if args.train_dir and args.test:
        # ── Mode: Train on multiple files, test on unseen ──
        print(f"\n--- TRAIN/TEST MODE (proper ML evaluation) ---")

        train_files = sorted(glob.glob(os.path.join(args.train_dir, "train_*.csv")))
        if not train_files:
            print(f"ERROR: No train_*.csv files in {args.train_dir}")
            sys.exit(1)

        print(f"\nTraining phase:")
        rewards = train_on_files(agent, train_files, args.episodes, args.threshold)

        print(f"\nTesting on unseen data: {args.test}")
        test_df = load_data(args.test)
        test_env = UPFScalingEnv(test_df["pps"].values, threshold=args.threshold)
        print(f"  {len(test_df)} samples, PPS range: "
              f"{test_df['pps'].min():.1f} — {test_df['pps'].max():.1f}")

    elif args.csv_file:
        # ── Mode: Single file (original behavior) ──
        print(f"\n--- SINGLE FILE MODE ---")
        print(f"\nLoading data from {args.csv_file}...")
        test_df = load_data(args.csv_file)
        pps_series = test_df["pps"].values
        print(f"  {len(test_df)} samples, PPS range: "
              f"{pps_series.min():.1f} — {pps_series.max():.1f}")

        env = UPFScalingEnv(pps_series, threshold=args.threshold)

        print(f"\nTraining for {args.episodes} episodes...")
        rewards = []
        for ep in range(args.episodes):
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
            rewards.append(total_reward)
            if ep % 10 == 0:
                agent.update_target()
            if (ep + 1) % 100 == 0:
                avg_r = np.mean(rewards[-100:])
                print(f"  Episode {ep+1}/{args.episodes} | "
                      f"Avg Reward: {avg_r:.1f} | Epsilon: {agent.epsilon:.3f}")

        test_env = UPFScalingEnv(pps_series, threshold=args.threshold)
    else:
        parser.print_help()
        sys.exit(1)

    # Evaluate
    print(f"\nEvaluating trained agent...")
    results = evaluate_agent(agent, test_env, test_df)
    metrics = compute_metrics(results)

    print(f"\n{'='*50}")
    print(f"  DQN Evaluation Results")
    print(f"{'='*50}")
    print(f"  Samples:                 {metrics['total_samples']}")
    print(f"  Match vs HPA:            {metrics['match_vs_hpa_pct']}%")
    print(f"  Match vs Ideal:          {metrics['match_vs_ideal_pct']}%")
    print(f"  Total reward:            {metrics['total_reward']}")
    print(f"  Avg reward/step:         {metrics['avg_reward']}")
    print(f"  DQN scale events:        {metrics['dqn_scale_events']}")
    print(f"  HPA scale events:        {metrics['hpa_scale_events']}")

    out_path = args.output or (args.test or args.csv_file).replace(".csv", "_dqn_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    curve_path = out_path.replace("_results.csv", "_training_curve.csv")
    pd.DataFrame({"episode": range(len(rewards)),
                   "total_reward": rewards}).to_csv(curve_path, index=False)
    print(f"  Training curve saved to: {curve_path}")


if __name__ == "__main__":
    main()
