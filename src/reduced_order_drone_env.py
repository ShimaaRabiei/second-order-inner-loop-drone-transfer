# -*- coding: utf-8 -*-
"""
Reduced-order drone environment used to train PPO policies for zero-shot
transfer to the second-order high-order drone model.

State/observation:
    o_k = [x_k, xdot_k, z_k, zdot_k]

Action:
    a_k = [Delta T_k, theta_star_k]

The action's second component is interpreted as the desired attitude reference
for the high-order deployment model.
"""

import os
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.callbacks import BaseCallback

# Windows/Anaconda OpenMP safety
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# Training/evaluation constants
# -----------------------------------------------------------------------------
N_ENVS = 8
TOTAL_BATCH = 8192
BATCH_SIZE = 512
N_STEPS = TOTAL_BATCH // N_ENVS
LEARNING_RATE = 4e-3
TOTAL_STEPS = 2_000_000
MAX_STEPS_EP = 400
REWARD_SCALE = 100.0
LAMBDA_PEN = 0.0
GAMMA = 0.994
SMOOTH_WIN = 10


class ReducedOrderDroneEnv(gym.Env):
    """Reduced translational drone model with direct attitude-reference action."""

    metadata = {"render.modes": ["console"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, -10], dtype=np.float32),
            high=np.array([20, 10, 20, 10], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-5.0, -np.pi / 8], dtype=np.float32),
            high=np.array([5.0, np.pi / 8], dtype=np.float32),
            dtype=np.float32,
        )
        self.g = 10.0
        self.m = 1.0
        self.dt = 0.05
        self.hover_T = 10.0
        self.max_steps = MAX_STEPS_EP
        self.state = None
        self.step_count = 0
        self.seed(SEED)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        x, z = self.np_random.choice(range(1, 10), size=2)
        vx, vz = self.np_random.uniform(-1, 1, size=2)
        self.state = np.array([x, vx, z, vz], dtype=np.float32)
        self.step_count = 0
        return self.state

    def step(self, action):
        x, x_dot, z, z_dot = self.state
        dT, theta_star = action

        T = np.clip(self.hover_T + dT, 0, 20)
        a_x = (T / self.m) * np.sin(theta_star)
        a_z = (T / self.m) * np.cos(theta_star) - self.g

        x_dot += a_x * self.dt
        z_dot += a_z * self.dt
        x += x_dot * self.dt
        z += z_dot * self.dt

        self.state = np.array([x, x_dot, z, z_dot], dtype=np.float32)

        d = np.array([x - 9.0, z - 9.0])
        reward = -float(np.dot(d, d)) / REWARD_SCALE
        if np.linalg.norm(d) < 0.1:
            reward += 10.0

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.state, reward, done, {}


class RewardCostWrapper(gym.Wrapper):
    """Adds the smoothness cost |Delta theta_star| to info and reward."""

    def __init__(self, env, lam=LAMBDA_PEN):
        super().__init__(env)
        self.prev_theta = 0.0
        self.lam = float(lam)

    def reset(self, **kwargs):
        self.prev_theta = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r_raw, done, info = self.env.step(action)
        cost = float(abs(action[1] - self.prev_theta))
        info.update({"cost": cost, "raw_r": r_raw})
        self.prev_theta = action[1]
        return obs, r_raw - self.lam * cost, done, info


def make_env(rank, lam=LAMBDA_PEN):
    def _init():
        env = RewardCostWrapper(ReducedOrderDroneEnv(), lam=lam)
        env.seed(SEED + rank)
        return env

    return _init


class ThreePanelLogger(BaseCallback):
    def __init__(self, gamma=GAMMA, lam=LAMBDA_PEN, window=SMOOTH_WIN):
        super().__init__()
        self.g = gamma
        self.lam = lam
        self.w = window
        self.R = []
        self.C = []
        self.RlamC = []
        self._g_r = 0.0
        self._g_c = 0.0
        self.t = 0

    def _on_step(self):
        info = self.locals["infos"][0]
        r_raw = info["raw_r"]
        cost = info["cost"]
        self._g_r += (self.g**self.t) * r_raw
        self._g_c += (self.g**self.t) * cost
        self.t += 1
        if self.locals["dones"][0]:
            self.R.append(self._g_r)
            self.C.append(self._g_c)
            self.RlamC.append(self._g_r - self.lam * self._g_c)
            self._g_r = 0.0
            self._g_c = 0.0
            self.t = 0
        return True

    def _panel(self, vals, title, ylabel):
        ep = np.arange(len(vals))
        plt.figure(figsize=(8, 4))
        plt.scatter(ep, vals, s=12, alpha=0.35)
        if len(vals) > self.w:
            ma = np.convolve(vals, np.ones(self.w) / self.w, "valid")
            plt.plot(ep[self.w - 1 :], ma, lw=2)
        plt.title(title)
        plt.xlabel("episode")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    def _on_training_end(self):
        self._panel(self.R, "gamma-discounted reward", r"$\sum \gamma^t r_t$")
        self._panel(self.C, "gamma-discounted |Delta theta| cost", r"$\sum \gamma^t |\Delta \theta_t|$")
        self._panel(
            self.RlamC,
            f"gamma-discounted reward - lambda cost, lambda={self.lam:g}",
            r"$\sum \gamma^t (r_t-\lambda |\Delta \theta_t|)$",
        )
