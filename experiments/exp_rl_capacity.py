#!/usr/bin/env python
# coding: utf-8
"""
Reinforcement Learning environment for interbank stability.
Optimizes a global haircut multiplier 'alfa' in [0, 1] using PPO.
"""
from pathlib import Path
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from interbank import Model
from interbank_lenderchange import LenderChange


class InterbankRLCapacityEnv(gym.Env):
    """Gymnasium environment where the agent chooses a global haircut scale alfa."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=1000, N=50):
        super().__init__()
        self.max_steps = max_steps
        self.N = N
        self.current_step = 0
        self.model = None

        # alfa scales the endogenous haircut; 1.0 keeps the model unchanged.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1e8, shape=(6,), dtype=np.float32)

    def _get_observation(self):
        active_mask = self.model.failed == 0
        reserves = self.model.C[active_mask]
        if len(reserves) == 0:
            reserves = np.array([0.0])

        C_max = float(np.max(reserves))
        C_min = float(np.min(reserves))
        C_avg = float(np.mean(reserves))

        ir_history = getattr(self.model.stats, "ir", [])
        r_avg = float(ir_history[-1]) if (len(ir_history) > 0 and not np.isnan(ir_history[-1])) else 0.0

        r_max_history = getattr(self.model.stats, "ir_max", [])
        r_max = float(r_max_history[-1]) if (len(r_max_history) > 0 and not np.isnan(r_max_history[-1])) else 0.0

        r_min_history = getattr(self.model.stats, "ir_min", [])
        r_min = float(r_min_history[-1]) if (len(r_min_history) > 0 and not np.isnan(r_min_history[-1])) else 0.0

        obs = np.array([C_max, C_min, C_avg, r_max, r_min, r_avg], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0)

    def _apply_global_haircut(self, alfa):
        self.model.haircut = np.where(
            self.model.d > 0,
            np.clip(alfa * self.model.haircut, 0.0, 1.0),
            np.nan,
        )
        self.model.capacity = np.where(
            self.model.d > 0,
            (1.0 - self.model.haircut) * self.model.d,
            np.nan,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.model = Model()
        self.model.configure(T=self.max_steps, N=self.N, seed=seed)
        self.model.lenderchange = LenderChange(self.model)
        self.model.stats.define_output_format("gdt")
        self.model.stats.generate_plots = False
        self.model.init()

        return self._get_observation(), {}

    def step(self, action):
        alfa = float(np.clip(action[0], 0.0, 1.0))

        t = self.current_step
        self.model.init_step(t)
        self.model.do_shock1()
        self.model.stats.compute_d1()
        self.model.stats.compute_potential_lenders()
        self.model.stats.compute_equity_lenders()
        self.model.stats.compute_equity_borrowers()
        self.model.setup_links()
        self.model.stats.compute_graph()
        self.model.do_interest_rate()
        self.model.stats.compute_psi()
        self.model.stats.compute_assets()

        # alfa only scales the haircut/capacity channel.
        self._apply_global_haircut(alfa)
        self.model.stats.compute_capacity()
        self.model.stats.compute_prob_bankruptcy()

        self.model.do_loans()
        self.model.stats.compute_ir()
        self.model.stats.compute_ir_weighted()
        self.model.stats.compute_rationing()
        self.model.stats.compute_num_loans()
        self.model.do_shock2()
        self.model.stats.compute_d2()

        profits_paid = self.model.do_repayments()
        self.model.stats.compute_profits(profits_paid)
        self.model.stats.compute_bankruptcies()
        self.model.stats.compute_borrower_failure_exposure()
        self.model.replace_failed_banks()
        self.model.stats.compute_num_banks()
        self.model.stats.compute_liquidity()
        self.model.stats.compute_deposits()
        self.model.stats.compute_reserves()
        self.model.stats.compute_bad_debt()
        self.model.stats.compute_equity()
        self.model.stats.compute_leverage()
        self.model.log.next()

        self.current_step += 1
        obs = self._get_observation()

        loans_this_step = self.model.stats.loans[-1] if len(self.model.stats.loans) > 0 else 0.0
        bankruptcies_this_step = self.model.stats.bankruptcies[-1] if len(self.model.stats.bankruptcies) > 0 else 0
        reward = float(loans_this_step - (bankruptcies_this_step * 200.0))

        terminated = False
        truncated = self.current_step >= self.max_steps

        active_banks = int(np.sum(self.model.failed == 0))
        if active_banks <= 2:
            terminated = True
            reward -= 10000.0

        equity_this_step = float(self.model.stats.equity[-1]) if len(self.model.stats.equity) > 0 else 0.0
        info = {"active_banks": active_banks, "alfa": alfa, "bankruptcies": bankruptcies_this_step, "loans": loans_this_step, "equity": equity_this_step}
        return obs, reward, terminated, truncated, info


def train_agent(timesteps=50000, model_path="ppo_optimal_alfa1"):
    env = InterbankRLCapacityEnv(max_steps=1000, N=50)
    print(f"Initializing PPO agent with MLP policy... Training for {timesteps} timesteps.")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        tensorboard_log=None,
    )

    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    print(f"Successfully trained and saved agent policy to '{model_path}'!")


def evaluate_and_compare(model_path="ppo_optimal_alfa1", steps=100):
    """Compare PPO policy against static alpha baselines."""
    env = InterbankRLCapacityEnv(max_steps=steps, N=50)

    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print("Trained RL policy loaded successfully.")
    else:
        print(f"Warning: Trained policy '{model_path}' not found. Please train first.")
        return

    obs, info = env.reset(seed=42)
    rl_equity = []
    rl_alfa_values = []
    rl_bankruptcies = []
    rl_loans = []
    rl_rewards = []

    for _ in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rl_equity.append(info["equity"])
        rl_alfa_values.append(info["alfa"])
        rl_bankruptcies.append(info["bankruptcies"])
        rl_loans.append(info["loans"])
        rl_rewards.append(reward)
        if terminated or truncated:
            break

    env.reset(seed=42)
    fixed_02_equity = []
    fixed_02_bankruptcies = []
    fixed_02_loans = []
    fixed_02_rewards = []
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step([0.2])
        fixed_02_equity.append(info["equity"])
        fixed_02_bankruptcies.append(info["bankruptcies"])
        fixed_02_loans.append(info["loans"])
        fixed_02_rewards.append(reward)
        if terminated or truncated:
            break

    env.reset(seed=42)
    fixed_08_equity = []
    fixed_08_bankruptcies = []
    fixed_08_loans = []
    fixed_08_rewards = []
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step([0.8])
        fixed_08_equity.append(info["equity"])
        fixed_08_bankruptcies.append(info["bankruptcies"])
        fixed_08_loans.append(info["loans"])
        fixed_08_rewards.append(reward)
        if terminated or truncated:
            break

    print("\n" + "=" * 50 + "\nSIMULATION SUMMARY:\n" + "=" * 50)
    print(f"RL Policy - Avg Equity: {np.mean(rl_equity):.2f}, Total Bankruptcies: {sum(rl_bankruptcies)}")
    print(f"Alpha = 0.2 - Avg Equity: {np.mean(fixed_02_equity):.2f}, Total Bankruptcies: {sum(fixed_02_bankruptcies)}")
    print(f"Alpha = 0.8 - Avg Equity: {np.mean(fixed_08_equity):.2f}, Total Bankruptcies: {sum(fixed_08_bankruptcies)}")

    results = [
        ["RL Policy", f"{np.mean(rl_equity):.2f}", str(sum(rl_bankruptcies)), f"{np.mean(rl_rewards):.2f}", f"{np.sum(rl_rewards):.2f}", f"{np.std(rl_rewards):.2f}", f"{np.mean(rl_loans):.2f}", f"{np.mean(rl_alfa_values):.2f}", str(len(rl_equity))],
        ["Alpha = 0.2", f"{np.mean(fixed_02_equity):.2f}", str(sum(fixed_02_bankruptcies)), f"{np.mean(fixed_02_rewards):.2f}", f"{np.sum(fixed_02_rewards):.2f}", f"{np.std(fixed_02_rewards):.2f}", f"{np.mean(fixed_02_loans):.2f}", "0.20", str(len(fixed_02_equity))],
        ["Alpha = 0.8", f"{np.mean(fixed_08_equity):.2f}", str(sum(fixed_08_bankruptcies)), f"{np.mean(fixed_08_rewards):.2f}", f"{np.sum(fixed_08_rewards):.2f}", f"{np.std(fixed_08_rewards):.2f}", f"{np.mean(fixed_08_loans):.2f}", "0.80", str(len(fixed_08_equity))],
    ]
    headers = ["Policy", "Avg Equity", "Total Bankruptcies", "Avg Reward", "Total Reward", "Std Reward", "Avg Loans/Step", "Avg Alfa", "Steps Executed"]
    widths = [
        max(len(headers[0]), max(len(row[0]) for row in results)),
        max(len(headers[1]), max(len(row[1]) for row in results)),
        max(len(headers[2]), max(len(row[2]) for row in results)),
        max(len(headers[3]), max(len(row[3]) for row in results)),
        max(len(headers[4]), max(len(row[4]) for row in results)),
        max(len(headers[5]), max(len(row[5]) for row in results)),
        max(len(headers[6]), max(len(row[6]) for row in results)),
        max(len(headers[7]), max(len(row[7]) for row in results)),
        max(len(headers[8]), max(len(row[8]) for row in results)),
    ]

    lines = []
    lines.append(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in results:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    os.makedirs("doc", exist_ok=True)
    results_path = os.path.join("doc", "rl_capacity1_results.txt")
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Results table saved successfully to '{results_path}'!")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(rl_equity, label="PPO RL Agent (Dynamic Alfa)")
    axes[0].plot(fixed_02_equity, label="Static Alfa = 0.2", linestyle="--")
    axes[0].plot(fixed_08_equity, label="Static Alfa = 0.8", linestyle=":")
    axes[0].set_title("Equity over Time")
    axes[0].set_ylabel("Equity")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(rl_bankruptcies, label="PPO RL Agent (Dynamic Alfa)")
    axes[1].plot(fixed_02_bankruptcies, label="Static Alfa = 0.2", linestyle="--")
    axes[1].plot(fixed_08_bankruptcies, label="Static Alfa = 0.8", linestyle=":")
    axes[1].set_title("Bankruptcies over Time")
    axes[1].set_xlabel("Simulation Timesteps")
    axes[1].set_ylabel("Bankruptcies")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("doc/rl_capacity1_comparison.png", dpi=300)
    plt.close()
    print("Comparison plot saved successfully to 'doc/rl_capacity1_comparison.png'!")

    total_equity = [np.nansum(rl_equity), np.nansum(fixed_02_equity), np.nansum(fixed_08_equity)]
    total_bankruptcies = [sum(rl_bankruptcies), sum(fixed_02_bankruptcies), sum(fixed_08_bankruptcies)]
    labels = ["PPO RL Agent", "Static Alfa = 0.2", "Static Alfa = 0.8"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, total_equity, color=["#4c78a8", "#f58518", "#54a24b"])
    plt.title("Total Equity Comparison")
    plt.ylabel("Total Equity")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("doc/comparison_equity_capacity1.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(labels, total_bankruptcies, color=["#4c78a8", "#f58518", "#54a24b"])
    plt.title("Total Bankruptcies Comparison")
    plt.ylabel("Total Bankruptcies")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("doc/comparison_bankruptcies_capacity1.png", dpi=300)
    plt.close()
    print("Comparison bar charts saved successfully to 'doc/comparison_equity_capacity1.png' and 'doc/comparison_bankruptcies_capacity1.png'!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reinforcement Learning for Interbank Network Stability via global haircut alfa in [0, 1]")
    parser.add_argument("--train", action="store_true", help="Train a new PPO agent")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the agent and compare to baselines")
    parser.add_argument("--steps", type=int, default=100, help="Steps for evaluation")
    parser.add_argument("--timesteps", type=int, default=10000, help="Timesteps for training")
    args = parser.parse_args()

    if args.train:
        train_agent(timesteps=args.timesteps)
    elif args.evaluate:
        evaluate_and_compare(steps=args.steps)
    else:
        parser.print_help()
