#!/usr/bin/env python3
"""
DQN-based Autoscaling Agent for 5G UPF.

Uses Deep Q-Network (reinforcement learning) to learn optimal
scaling decisions from recorded PPS data.

State:  [current_pps, pps_trend, current_replicas]
Action: 0=scale_down, 1=hold, 2=scale_up
Reward: - penalty for wrong replica count vs ideal
        - penalty for unnecessary scaling (oscillation)
        - bonus for matching ideal replicas

Usage:
    python dqn_predictor.py <csv_file> [--episodes N] [--threshold T]
"""

import argparse
import sys
import os
import warnings
import random
from collections import deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Check for PyTorch ───
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ══════════════════════════════════════════════════════════════
#  Environment: simulates UPF scaling from recorded PPS data
# ══════════════════════════════════════════════════════════════
class UPFScalingEnv:
    """
    Gym-like environment that replays recorded PPS data.
    The agent decides: scale_down (0), hold (1), scale_up (2)
    """

    def __init__(self, pps_series, threshold=4000, min_replicas=1, max_replicas=5):
        self.pps = pps_series  # array of PPS values
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
        """State: [normalized_pps, pps_trend, normalized_replicas]"""
        pps_now = self.pps[self.t] / self.threshold  # normalize around threshold
        if self.t > 0:
            pps_trend = (self.pps[self.t] - self.pps[self.t - 1]) / self.threshold
        else:
            pps_trend = 0.0
        replicas_norm = self.replicas / self.max_replicas
        return np.array([pps_now, pps_trend, replicas_norm], dtype=np.float32)

    def _ideal_replicas(self, pps):
        """What replicas SHOULD be for this PPS level."""
        if pps <= 0:
            return 1
        return max(1, min(int(np.ceil(pps / self.threshold)), self.max_replicas))

    def step(self, action):
        """
        action: 0=scale_down, 1=hold, 2=scale_up
        Returns: (next_state, reward, done)
        """
        self.prev_replicas = self.replicas

        # Apply action
        if action == 0:  # scale down
            self.replicas = max(self.min_replicas, self.replicas - 1)
        elif action == 2:  # scale up
            self.replicas = min(self.max_replicas, self.replicas + 1)
        # action == 1: hold

        # Calculate reward
        ideal = self._ideal_replicas(self.pps[self.t])
        reward = 0.0

        # Reward for correct replicas
        diff = abs(self.replicas - ideal)
        if diff == 0:
            reward += 10.0   # perfect match
        else:
            reward -= diff * 5.0  # penalty proportional to mismatch

        # Penalty for unnecessary scaling (oscillation)
        if self.replicas != self.prev_replicas:
            if self.replicas != ideal:
                reward -= 3.0  # penalize wrong scaling action
            else:
                reward += 2.0  # small bonus for correct scaling action

        # Penalty for over-provisioning (wasting resources)
        if self.replicas > ideal:
            reward -= (self.replicas - ideal) * 2.0

        # Penalty for under-provisioning (SLA violation risk)
        if self.replicas < ideal:
            reward -= (ideal - self.replicas) * 3.0

        # Advance time
        self.t += 1
        done = self.t >= self.n_steps

        next_state = self._get_state() if not done else np.zeros(3, dtype=np.float32)
        return next_state, reward, done


# ══════════════════════════════════════════════════════════════
#  DQN Neural Network
# ══════════════════════════════════════════════════════════════
if HAS_TORCH:
    class DQN(nn.Module):
        def __init__(self, state_size=3, action_size=3, hidden=64):
            super(DQN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_size),
            )

        def forward(self, x):
            return self.net(x)


# ══════════════════════════════════════════════════════════════
#  DQN Agent with Experience Replay
# ══════════════════════════════════════════════════════════════
class DQNAgent:
    def __init__(self, state_size=3, action_size=3, lr=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, memory_size=2000, batch_size=32):
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
            # Fallback: simple Q-table approximation
            self.q_table = {}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        if HAS_TORCH:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item()
        else:
            key = tuple(np.round(state, 2))
            if key not in self.q_table:
                return random.randrange(self.action_size)
            return int(np.argmax(self.q_table[key]))

    def replay(self):
        """Train on a batch from experience replay memory."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)

        if HAS_TORCH:
            states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
            actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
            next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
            dones = torch.BoolTensor([b[4] for b in batch]).to(self.device)

            # Current Q values
            q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

            # Target Q values
            with torch.no_grad():
                q_next = self.target_net(next_states).max(dim=1)[0]
                q_next[dones] = 0.0
                q_target = rewards + self.gamma * q_next

            loss = self.loss_fn(q_current, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        else:
            # Fallback Q-table update
            for s, a, r, ns, d in batch:
                key = tuple(np.round(s, 2))
                nkey = tuple(np.round(ns, 2))
                if key not in self.q_table:
                    self.q_table[key] = np.zeros(self.action_size)
                if nkey not in self.q_table:
                    self.q_table[nkey] = np.zeros(self.action_size)
                target = r if d else r + self.gamma * np.max(self.q_table[nkey])
                self.q_table[key][a] += 0.1 * (target - self.q_table[key][a])
            return 0.0

    def update_target(self):
        """Copy policy network weights to target network."""
        if HAS_TORCH:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ══════════════════════════════════════════════════════════════
#  Data Loading & Evaluation
# ══════════════════════════════════════════════════════════════
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts_iso"])
    df = df.set_index("ts").sort_index()
    df["pps"] = pd.to_numeric(df["pps_prom"], errors="coerce").fillna(0)
    df["replicas"] = pd.to_numeric(df["hpa_replicas"], errors="coerce").fillna(1).astype(int)
    return df


def evaluate_agent(agent, env, df):
    """Run trained agent through the data and collect predictions."""
    state = env.reset()
    agent.epsilon = 0.0  # no exploration during evaluation

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
    valid = results_df.dropna()
    return {
        "total_samples": len(valid),
        "match_vs_hpa_pct": round(valid["replica_match_hpa"].mean() * 100, 1),
        "match_vs_ideal_pct": round(valid["replica_match_ideal"].mean() * 100, 1),
        "total_reward": round(valid["reward"].sum(), 2),
        "avg_reward": round(valid["reward"].mean(), 2),
        "dqn_scale_events": int((valid["dqn_replicas"].diff().fillna(0) != 0).sum()),
        "hpa_scale_events": int((valid["actual_replicas"].diff().fillna(0) != 0).sum()),
    }


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="DQN Autoscaling Agent for 5G UPF")
    parser.add_argument("csv_file", help="Path to watcher CSV")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Training episodes (default: 500)")
    parser.add_argument("--threshold", type=int, default=4000,
                        help="PPS threshold per replica (default: 4000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    print(f"DQN Autoscaling Agent")
    print(f"{'='*50}")
    print(f"Backend: {'PyTorch' if HAS_TORCH else 'Q-Table (fallback)'}")

    print(f"\nLoading data from {args.csv_file}...")
    df = load_data(args.csv_file)
    pps_series = df["pps"].values
    print(f"  {len(df)} samples, PPS range: {pps_series.min():.1f} — {pps_series.max():.1f}")

    # Create environment and agent
    env = UPFScalingEnv(pps_series, threshold=args.threshold)
    agent = DQNAgent(state_size=3, action_size=3)

    # Training
    print(f"\nTraining for {args.episodes} episodes...")
    rewards_history = []
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
        rewards_history.append(total_reward)

        # Update target network every 10 episodes
        if ep % 10 == 0:
            agent.update_target()

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(f"  Episode {ep+1}/{args.episodes} | "
                  f"Avg Reward: {avg_r:.1f} | Epsilon: {agent.epsilon:.3f}")

    # Evaluation
    print(f"\nEvaluating trained agent...")
    env_eval = UPFScalingEnv(pps_series, threshold=args.threshold)
    results = evaluate_agent(agent, env_eval, df)
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

    # Save results
    out_path = args.output or args.csv_file.replace(".csv", "_dqn_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    # Save training curve
    curve_path = out_path.replace("_results.csv", "_training_curve.csv")
    pd.DataFrame({"episode": range(len(rewards_history)),
                   "total_reward": rewards_history}).to_csv(curve_path, index=False)
    print(f"  Training curve saved to: {curve_path}")


if __name__ == "__main__":
    main()
