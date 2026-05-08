# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:52:58 2025

@author: RABIEI
"""

# -*- coding: utf-8 -*-
"""

H_early_sum_mmer_2nd_theta.py  (UPDATED: 2nd-order inner loop)

---------------------------------------------------------------------
Requirements (conda env rl-cpu):
    numpy scipy matplotlib pillow gym shimmy
    pytorch torchvision torchaudio pytorch-cuda=12.1
    stable-baselines3[extra]  control
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random, math
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import control as ctrl  # python-control (continuous <-> discrete)

# =============================================================================
# Reproducibility
# =============================================================================
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# RL hyper-parameters 
# =============================================================================
N_ENVS        = 8
TOTAL_BATCH   = 8192
BATCH_SIZE    = 512
N_STEPS       = TOTAL_BATCH // N_ENVS
LEARNING_RATE = 4e-3
TOTAL_STEPS   = 2_000_000
MAX_STEPS_EP  = 400

REWARD_SCALE  = 1e2   #
LAMBDA_PEN    = 0.0
GAMMA         = 0.994
SMOOTH_WIN    = 10

DELTA_T       = 0.05  # control/simulation timestep

# =============================================================================
# Inner-loop parameters (2nd order)
# =============================================================================
INERTIA_J     = 0.02  #

# Option A (recommended): specify wn and zeta (then Kp, Kd come from them)
WN_DEFAULT    = 2.0
ZETA_DEFAULT  = 0.4

# Option B: specify Kp and (optionally) Kd directly
Kp_DEFAULT    = None   #
Kd_DEFAULT    = None   #

# θ★ derivative estimator
THETA_STAR_TAU = 0.1   # [s] low-pass time constant for θ̇★ estimate


def gains_from_specs(wn: float, zeta: float, J: float):
    """Given wn,zeta for 2nd-order: Kp = J*wn^2, Kd = 2*J*zeta*wn."""
    Kp = J * (wn ** 2)
    Kd = 2.0 * J * zeta * wn
    return Kp, Kd


def specs_from_gains(Kp: float, Kd: float, J: float):
    wn = math.sqrt(max(Kp, 0.0) / J) if (Kp is not None and Kp > 0) else 0.0
    zeta = (Kd / (2.0 * J * wn)) if wn > 0 else float("nan")
    return wn, zeta


# =============================================================================
# 1) HIGH-order environment (6-state) with SECOND-order inner loop
# =============================================================================
class HighOrderDroneEnv(gym.Env):
    """
    state  = [x, ẋ, z, ż, θ, θ̇]
    action = (ΔT, θ★)

    Outer (translation) dynamics (Euler @ delta_t):
        ax = (T/m) sin(θ)
        az = (T/m) cos(θ) - g

    Inner θ dynamics (2nd-order PD with inertia J):
        J θ̈ = -Kp (θ - θ★) - Kd (θ̇ - θ̇★_est)

    θ̇★_est is estimated from θ★ by backward difference at delta_t and
    low-pass filtered:
        raw = (θ★_k - θ★_{k-1}) / delta_t
        θ̇★_est <- alpha*θ̇★_est + (1-alpha)*raw
        alpha = exp(-delta_t/tau)
    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        delta_t=DELTA_T,
        J=INERTIA_J,
        # Choose ONE parameterization:
        wn=WN_DEFAULT,
        zeta=ZETA_DEFAULT,
        Kp=Kp_DEFAULT,
        Kd=Kd_DEFAULT,
        theta_star_tau=THETA_STAR_TAU,
        print_specs_once=False,
    ):
        super().__init__()

        self.delta_t = float(delta_t)
        self.J = float(J)

        # Outer constants
        self.g, self.m = 10.0, 1.0
        self.hover_T = self.g * self.m
        self.max_steps = MAX_STEPS_EP

        # ------------------ choose inner-loop gains ------------------
        #
        #
        self.wn = float(wn) if (wn is not None) else None
        self.zeta = float(zeta) if (zeta is not None) else None

        if (self.wn is not None) and (self.zeta is not None):
            self.Kp, self.Kd = gains_from_specs(self.wn, self.zeta, self.J)
        else:
            self.Kp = float(Kp) if Kp is not None else 0.0
            if Kd is None:
                # critical damping-ish heuristic
                self.Kd = 2.0 * math.sqrt(max(self.J * self.Kp, 0.0))
            else:
                self.Kd = float(Kd)
            self.wn, self.zeta = specs_from_gains(self.Kp, self.Kd, self.J)

        # ------------------ θ★ derivative estimator ------------------
        self.theta_star_prev = 0.0
        self.theta_star_dot_est = 0.0
        tau = float(theta_star_tau)
        self.theta_star_alpha = math.exp(-self.delta_t / tau) if tau > 0 else 0.0

        # Build discrete inner-loop model at delta_t
        self._build_state_space()

        # Observation and action spaces
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-10.0, -np.pi / 8], np.float32),
            high=np.array([10.0, np.pi / 8], np.float32),
        )

        self.state = None
        self.step_count = 0
        self.seed()

        if print_specs_once:
            print("\n[HighOrderDroneEnv] 2nd-order inner loop:")
            print(f"  J    = {self.J:.6f}")
            print(f"  wn   = {self.wn:.3f} rad/s")
            print(f"  zeta = {self.zeta:.3f}")
            print(f"  Kp   = {self.Kp:.3f}")
            print(f"  Kd   = {self.Kd:.3f}")
            print("[HighOrderDroneEnv] A_c =\n", self.A_c)
            print("[HighOrderDroneEnv] B_c =\n", self.B_c)

    def _build_state_space(self):
        """
        Continuous-time inner-loop:
            xθ = [θ, θ̇]^T
            u  = [θ★, θ̇★_est]^T

            ẋ = A_c x + B_c u
            A_c = [[0, 1],
                   [-Kp/J, -Kd/J]]
            B_c = [[0, 0],
                   [Kp/J, Kd/J]]

        Discretize at delta_t using ZOH.
        """
        self.A_c = np.array(
            [
                [0.0, 1.0],
                [-(self.Kp / self.J), -(self.Kd / self.J)],
            ],
            dtype=float,
        )
        self.B_c = np.array(
            [
                [0.0, 0.0],
                [self.Kp / self.J, self.Kd / self.J],
            ],
            dtype=float,
        )
        self.C_c = np.array([[1.0, 0.0]], dtype=float)
        self.D_c = np.array([[0.0, 0.0]], dtype=float)

        sys_c = ctrl.ss(self.A_c, self.B_c, self.C_c, self.D_c)

        # robust discretization across python-control versions
        try:
            sys_d = ctrl.sample_system(sys_c, self.delta_t, method="zoh")
        except AttributeError:
            sys_d = ctrl.c2d(sys_c, self.delta_t, method="zoh")

        self.Ad, self.Bd = sys_d.A, sys_d.B  # Ad: 2x2, Bd: 2x2

    # ---------------- Gym API ----------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        x, z = self.np_random.choice(range(1, 10), size=2)
        vx, vz = self.np_random.uniform(-1, 1, size=2)
        theta, theta_dot = 0.0, 0.0

        self.state = np.array([x, vx, z, vz, theta, theta_dot], np.float32)
        self.step_count = 0

        # reset θ★ derivative estimator
        self.theta_star_prev = 0.0
        self.theta_star_dot_est = 0.0

        return self.state

    def step(self, action):
        # unpack state & action
        x, vx, z, vz, theta, theta_dot = map(float, self.state)
        dT, theta_star = map(float, action)

        # thrust
        T = self.hover_T + dT
        #
        # T = np.clip(T, 0.0, 20.0)

        # ----- θ̇★ estimate (backward diff @ delta_t + low-pass) -----
        if self.step_count == 0:
            raw_theta_star_dot = 0.0
        else:
            raw_theta_star_dot = (theta_star - self.theta_star_prev) / self.delta_t

        alpha = self.theta_star_alpha
        self.theta_star_dot_est = (
            alpha * self.theta_star_dot_est
            + (1.0 - alpha) * raw_theta_star_dot
        )
        self.theta_star_prev = theta_star

        # ----- inner-loop discrete update (single step @ delta_t) -----
        th_vec = np.array([[theta], [theta_dot]], dtype=float)                  # 2x1
        u_vec  = np.array([[theta_star], [self.theta_star_dot_est]], dtype=float)  # 2x1
        th_next = self.Ad @ th_vec + self.Bd @ u_vec
        theta_next, theta_dot_next = th_next.ravel()

        # ----- translational dynamics (Euler @ delta_t) -----
        ax = (T / self.m) * math.sin(theta_next)
        az = (T / self.m) * math.cos(theta_next) - self.g

        vx += ax * self.delta_t
        vz += az * self.delta_t
        x  += vx * self.delta_t
        z  += vz * self.delta_t

        self.state = np.array([x, vx, z, vz, theta_next, theta_dot_next], np.float32)

        # ----- reward  -----
        d = np.array([x - 9.0, z - 9.0], np.float32)
        reward = -float(d @ d) / REWARD_SCALE
        if np.linalg.norm(d) < 0.1:
            reward += 10.0

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.state, reward, done, {}


# =============================================================================
# 2) Reward-cost wrapper 
# =============================================================================
class RewardCostWrapper(gym.Wrapper):
    def __init__(self, env, lam=LAMBDA_PEN):
        super().__init__(env)
        self.prev_theta_star = 0.0
        self.lam = float(lam)

    def reset(self, **kw):
        self.prev_theta_star = 0.0
        return self.env.reset(**kw)

    def step(self, action):
        obs, r_raw, done, info = self.env.step(action)
        cost = float(abs(action[1] - self.prev_theta_star))
        info = dict(info)
        info.update({"cost": cost, "raw_r": r_raw})
        self.prev_theta_star = float(action[1])
        return obs, r_raw - self.lam * cost, done, info


# =============================================================================
# 3) Reduced Obs/Action adapter (agent sees 4-D obs: [x, ẋ, z, ż])
# =============================================================================
class ReducedObsActionAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low[:4]
        high = env.observation_space.high[:4]
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kw):
        full = self.env.reset(**kw)
        return full[:4]

    def step(self, agent_act):
        full_next, rew, done, info = self.env.step(agent_act)
        return full_next[:4], rew, done, info


def make_env(rank):
    def _init():
        env = HighOrderDroneEnv(
            delta_t=DELTA_T,
            J=INERTIA_J,
            wn=WN_DEFAULT,
            zeta=ZETA_DEFAULT,
            # If want to override with Kp/Kd instead, set wn=None,zeta=None and pass Kp/Kd
            # wn=None, zeta=None, Kp=10.0, Kd=None,
            print_specs_once=(rank == 0),
        )
        env = RewardCostWrapper(env, lam=LAMBDA_PEN)
        env = ReducedObsActionAdapter(env)
        env.seed(SEED + rank)
        return env
    return _init


# =============================================================================
# 4) Logger (same 3-panel logger style)
# =============================================================================
class ThreePanelLogger(BaseCallback):
    def __init__(self, gamma=GAMMA, lam=LAMBDA_PEN, window=SMOOTH_WIN):
        super().__init__()
        self.g, self.lam, self.w = float(gamma), float(lam), int(window)
        self.R, self.C, self.RlamC = [], [], []
        self._g_r = 0.0
        self._g_c = 0.0
        self.t = 0

    def _on_step(self):
        info = self.locals["infos"][0]
        r_raw, cost = info["raw_r"], info["cost"]
        self._g_r += (self.g ** self.t) * float(r_raw)
        self._g_c += (self.g ** self.t) * float(cost)
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
        self._panel(self.R, "γ-discounted reward", r"$\sum γ^t r_t$")
        self._panel(self.C, "γ-discounted |Δθ★| cost", r"$\sum γ^t |Δθ★_t|$")
        self._panel(self.RlamC, f"γ-discounted (r-λc), λ={self.lam}", r"$\sum γ^t(r_t-λ|Δθ★_t|)$")


# =============================================================================
# 5) Training script
# =============================================================================
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecNormalize(VecMonitor(vec_env), norm_obs=True, norm_reward=True, gamma=GAMMA)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=0.98,
        clip_range=0.05,
        vf_coef=0.8,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        device="cpu",
        seed=SEED,
        verbose=1,
    )

    RUN_ID = "HighOrder_2ndOrderInnerLoop_NoSubstep"
    model.learn(total_timesteps=TOTAL_STEPS, callback=ThreePanelLogger())
    model.save(RUN_ID)
    vec_env.save(f"_{RUN_ID}.pkl")
