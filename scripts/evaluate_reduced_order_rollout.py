# -*- coding: utf-8 -*-
"""Evaluate one deterministic rollout of a reduced-order drone PPO policy."""

import argparse
import sys
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "models" / "lambda_policies"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reduced_order_drone_env import (  # noqa: E402
    BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    MAX_STEPS_EP,
    N_ENVS,
    N_STEPS,
    REWARD_SCALE,
    SMOOTH_WIN,
    TOTAL_BATCH,
    TOTAL_STEPS,
    make_env,
)


class FixedStart(gym.Wrapper):
    """On the first reset, force env.state to a chosen reduced state."""

    def __init__(self, env, start):
        super().__init__(env)
        self.start = np.asarray(start, dtype=np.float32)
        self.first = True

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.first:
            inner = self.env
            while isinstance(inner, gym.Wrapper):
                inner = inner.env
            inner.state = self.start.copy()
            obs = self.start.copy()
            self.first = False
        return obs


def lambda_tag(lam: float) -> str:
    return f"lambda_{lam:g}"


def to_flat(x):
    return np.asarray(x).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate one reduced-order policy rollout.")
    parser.add_argument("--lambda_pen", type=float, default=0.0, help="Penalty weight used by the policy.")
    parser.add_argument("--model", type=str, default="", help="Optional explicit PPO model path.")
    parser.add_argument("--vecnormalize", type=str, default="", help="Optional explicit VecNormalize path.")
    parser.add_argument("--random_start", action="store_true", help="Use the environment random reset instead of fixed start.")
    parser.add_argument("--fixed_state", type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0])
    args = parser.parse_args()

    lam = float(args.lambda_pen)
    tag = lambda_tag(lam)

    model_file = Path(args.model) if args.model else MODEL_DIR / f"{tag}_policy.zip"
    vec_file = Path(args.vecnormalize) if args.vecnormalize else MODEL_DIR / f"{tag}_vecnormalize.pkl"

    if args.random_start:
        raw_env = DummyVecEnv([make_env(0, lam=lam)])
    else:
        fixed_state = np.array(args.fixed_state, dtype=np.float32)
        raw_env = DummyVecEnv([lambda: FixedStart(make_env(0, lam=lam)(), fixed_state)])

    eval_env = VecNormalize.load(str(vec_file), raw_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(str(model_file), env=eval_env, device="cpu")

    obs = eval_env.reset()
    xs, zs, theta_stars, ts = [], [], [], []
    raw_rewards, costs = [], []
    cumulative_normalized_return = 0.0
    cumulative_cost = 0.0

    for t in range(MAX_STEPS_EP):
        real = to_flat(eval_env.unnormalize_obs(obs))
        xs.append(real[0])
        zs.append(real[2])
        ts.append(t)

        action, _ = model.predict(obs, deterministic=True)
        theta_stars.append(float(to_flat(action)[1]))

        obs, r, done, info = eval_env.step(action)
        raw_rewards.append(info[0]["raw_r"])
        costs.append(info[0]["cost"])
        cumulative_normalized_return += float(to_flat(r)[0])
        cumulative_cost += float(info[0]["cost"])
        if done:
            break

    gamma = model.gamma
    powers = gamma ** np.arange(len(raw_rewards))
    disc_reward = float(np.dot(powers, raw_rewards))
    disc_cost = float(np.dot(powers, costs))
    disc_return = disc_reward - lam * disc_cost

    summary = (
        f"gamma-discounted return: {disc_return:.3f}   "
        f"gamma-discounted reward: {disc_reward:.3f}   "
        f"gamma-discounted |Delta theta|: {disc_cost:.3f}"
    )
    print(f"\nModel: {model_file}")
    print(f"VecNormalize: {vec_file}")
    print(f"Undiscounted normalized episode return: {cumulative_normalized_return:.4f}")
    print(f"Undiscounted cumulative |Delta theta|: {cumulative_cost:.4f}")
    print(summary)

    plt.figure(figsize=(5, 5))
    plt.plot(xs, zs, "-o", ms=2, label="path")
    plt.scatter(9, 9, label="target")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.gca().set_aspect("equal")
    plt.title("Reduced-order deterministic policy trajectory")
    plt.legend()
    plt.gcf().text(0.5, 0.97, summary, ha="center", va="top", fontsize=9)
    plt.tight_layout()
    plt.show()

    for values, ylabel, title in [
        (xs, "x [m]", "X position over episode"),
        (zs, "z [m]", "Z position over episode"),
        (theta_stars, "theta_star [rad]", "Desired attitude reference over episode"),
    ]:
        plt.figure(figsize=(6, 3))
        plt.plot(ts, values)
        plt.xlabel("timestep")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    rows = [
        ("gamma-discounted return", f"{disc_return:.3f}"),
        ("gamma-discounted reward", f"{disc_reward:.3f}"),
        ("gamma-discounted |Delta theta|", f"{disc_cost:.3f}"),
        ("N_ENVS", N_ENVS),
        ("TOTAL_BATCH", TOTAL_BATCH),
        ("BATCH_SIZE", BATCH_SIZE),
        ("N_STEPS", N_STEPS),
        ("LEARNING_RATE", LEARNING_RATE),
        ("TOTAL_STEPS", TOTAL_STEPS),
        ("MAX_STEPS_EP", MAX_STEPS_EP),
        ("REWARD_SCALE", REWARD_SCALE),
        ("lambda_pen", lam),
        ("GAMMA", GAMMA),
        ("SMOOTH_WIN", SMOOTH_WIN),
        ("gae_lambda", model.gae_lambda),
        ("clip_range", model.clip_range if not callable(model.clip_range) else model.clip_range(1.0)),
        ("vf_coef", model.vf_coef),
    ]
    df = pd.DataFrame(rows, columns=["Parameter", "Value"])
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
