# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:59:34 2025

@author: RABIEI
"""

# -*- coding: utf-8 -*-
"""
HE_early_summer_2nd_theta_E.py

Evaluate ONE deterministic episode of a trained PPO policy on the *high-order*
drone environment (6-state internally, 4-D obs to PPO) where the INNER θ-loop is
SECOND ORDER (J, Kp, Kd, theta_star_dot_est filter), matching the UPDATED
ppo_train_high_order.py I gave you.

Main edits vs your old evaluator:
- No "kp_val" argument into make_env (because we rebuilt env here directly).
- Shows J, Kp, Kd (and wn,zeta if used) on plots.
- Tries both RUN_ID.pkl and _RUN_ID.pkl for VecNormalize stats.

Edit CODE_DIR, RUN_ID, and (optionally) the inner-loop parameters below.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---- file & run settings -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "models" / "lambda_policies"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RUN_ID = "lambda_0"

MODEL_FILE = MODEL_DIR / "lambda_0_policy.zip"
VECNORMALIZE_FILE = MODEL_DIR / "lambda_0_vecnormalize.pkl"

USE_FIXED_START = True
FIXED_STATE = np.array([1., 1., 1., 1.], np.float32)  # [x, xdot, z, zdot]

# ---- import shared stuff from training file ------------------------------
from env_second_order_inner_loop import (
    HighOrderDroneEnv,
    RewardCostWrapper,
    ReducedObsActionAdapter,
    MAX_STEPS_EP,
    LAMBDA_PEN,
    N_ENVS, TOTAL_BATCH, BATCH_SIZE, N_STEPS, LEARNING_RATE,
    TOTAL_STEPS, REWARD_SCALE, GAMMA, SMOOTH_WIN,
    DELTA_T, INERTIA_J, WN_DEFAULT, ZETA_DEFAULT, THETA_STAR_TAU
)

# =============================================================================
# Helper wrapper to force the first reset state (same idea as your old code)
# =============================================================================
class FixedStart(gym.Wrapper):
    def __init__(self, env, start):
        super().__init__(env)
        self._start = start
        self._first = True

    def reset(self, **kw):
        obs = self.env.reset(**kw)
        if self._first:
            inner = self.env
            while isinstance(inner, gym.Wrapper):
                inner = inner.env
            # inner is the base HighOrderDroneEnv
            inner.state[:4] = self._start.copy()
            obs = self._start.copy()
            self._first = False
        return obs


# =============================================================================
# Build the eval environment (MATCH the wrappers used in training)
# =============================================================================
# Choose ONE parameterization for the inner loop:
USE_WN_ZETA = True

# If USE_WN_ZETA=True:
WN   = WN_DEFAULT
ZETA = ZETA_DEFAULT

# If USE_WN_ZETA=False:
KP = None     # e.g., 10.0
KD = None     # None -> critical damping-ish formula inside env

# You can also override inertia J and filter tau here:
J_EVAL   = INERTIA_J
TAU_EVAL = THETA_STAR_TAU

def build_eval_env(print_specs=False):
    if USE_WN_ZETA:
        base = HighOrderDroneEnv(
            delta_t=DELTA_T,
            J=J_EVAL,
            wn=WN,
            zeta=ZETA,
            theta_star_tau=TAU_EVAL,
            print_specs_once=print_specs,
        )
    else:
        base = HighOrderDroneEnv(
            delta_t=DELTA_T,
            J=J_EVAL,
            wn=None, zeta=None,
            Kp=KP, Kd=KD,
            theta_star_tau=TAU_EVAL,
            print_specs_once=print_specs,
        )

    env = RewardCostWrapper(base, lam=LAMBDA_PEN)
    env = ReducedObsActionAdapter(env)  # PPO sees 4-D obs
    env.seed(0)
    return env


# =============================================================================
# Load VecNormalize stats
# =============================================================================
vn_path = VECNORMALIZE_FILE
if not vn_path.exists():
    raise FileNotFoundError(f"VecNormalize .pkl not found: {vn_path}")

# DummyVecEnv must wrap an env factory
def make_one_env():
    env = build_eval_env(print_specs=True)
    if USE_FIXED_START:
        env = FixedStart(env, FIXED_STATE)
    return env

raw_env = DummyVecEnv([make_one_env])

eval_env = VecNormalize.load(str(vn_path), venv=raw_env)
eval_env.training = False
eval_env.norm_reward = False

# =============================================================================
# Load trained policy
# =============================================================================
model_path = str(MODEL_FILE)
model = PPO.load(model_path, env=eval_env, device="cpu")

# =============================================================================
# Read actual inner-loop params in use (unwrap to base env)
# =============================================================================
base_env = eval_env.venv.envs[0]
while isinstance(base_env, gym.Wrapper):
    base_env = base_env.env

J_USED   = float(getattr(base_env, "J", np.nan))
KP_USED  = float(getattr(base_env, "Kp", np.nan))
KD_USED  = float(getattr(base_env, "Kd", np.nan))
WN_USED  = float(getattr(base_env, "wn", np.nan))
ZETA_USED= float(getattr(base_env, "zeta", np.nan))

# =============================================================================
# Run ONE deterministic episode
# =============================================================================
obs = eval_env.reset()

xs, zs, ts = [], [], []
theta_stars = []
theta_actuals, theta_dots = [], []

raw_rs, costs = [], []
cum_r, cum_c = 0.0, 0.0

to_flat = lambda x: np.asarray(x).reshape(-1)

for t in range(MAX_STEPS_EP):
    real = to_flat(eval_env.unnormalize_obs(obs))
    xs.append(real[0])
    zs.append(real[2])
    ts.append(t)

    # also log the *actual* theta, theta_dot from the base env state
    theta_actuals.append(float(base_env.state[4]))
    theta_dots.append(float(base_env.state[5]))

    action, _ = model.predict(obs, deterministic=True)
    theta_stars.append(float(to_flat(action)[1]))

    obs, r_step, done, info = eval_env.step(action)

    raw_rs.append(float(info[0]["raw_r"]))
    costs.append(float(info[0]["cost"]))
    cum_r += float(to_flat(r_step)[0])
    cum_c += float(info[0]["cost"])

    if done:
        break

print(f"\nUndisc. episode return : {cum_r:.3f}")
print(f"Undisc. Σ|Δθ★|         : {cum_c:.3f}")

# ---- γ-discounted metrics -----------------------------------------------
g = model.gamma
pw = g ** np.arange(len(raw_rs))
disc_r   = float(pw @ np.asarray(raw_rs))
disc_c   = float(pw @ np.asarray(costs))
disc_ret = disc_r - LAMBDA_PEN * disc_c

summary = (
    f"J={J_USED:.3f}  Kp={KP_USED:.2f}  Kd={KD_USED:.2f}   "
    f"(wn={WN_USED:.2f}, zeta={ZETA_USED:.2f})   "
    u"\u03B3-disc return: " f"{disc_ret:.3f}   "
    u"\u03B3-disc reward: " f"{disc_r:.3f}   "
    u"\u03B3-disc |Δ\u03B8★|: " f"{disc_c:.3f}"
)
print(summary)

def stamp():
    plt.gcf().text(
        0.5, 0.97, summary,
        ha="center", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, pad=2)
    )

# =============================================================================
# Plots
# =============================================================================
# Trajectory
plt.figure(figsize=(5, 5))
plt.plot(xs, zs, "-o", ms=2)
plt.scatter(9, 9, c="red", label="target")
plt.xlim(0, 20); plt.ylim(0, 20)
plt.gca().set_aspect("equal")
plt.title("Trajectory in x-z plane")
plt.legend()
stamp()
plt.show()

# Time-series
for arr, lbl in [
    (xs, "x(t) [m]"),
    (zs, "z(t) [m]"),
    (theta_stars, "θ★(t) [rad]"),
    (theta_actuals, "θ(t) actual [rad]"),
    (theta_dots, "θ̇(t) actual [rad/s]"),
]:
    plt.figure(figsize=(6, 3))
    plt.plot(ts, arr)
    plt.xlabel("timestep")
    plt.ylabel(lbl)
    plt.title(lbl)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Summary table
# =============================================================================
rows = [
    ("J",              J_USED),
    ("Kp",             KP_USED),
    ("Kd",             KD_USED),
    ("wn",             WN_USED),
    ("zeta",           ZETA_USED),
    ("tau (θ★ LPF)",   TAU_EVAL),

    ("γ-disc return",  f"{disc_ret:.3f}"),
    ("γ-disc reward",  f"{disc_r:.3f}"),
    ("γ-disc |Δθ★|",    f"{disc_c:.3f}"),

    ("N_ENVS",       N_ENVS),
    ("TOTAL_BATCH",  TOTAL_BATCH),
    ("BATCH_SIZE",   BATCH_SIZE),
    ("N_STEPS",      N_STEPS),
    ("LEARNING_RATE",LEARNING_RATE),
    ("TOTAL_STEPS",  TOTAL_STEPS),
    ("MAX_STEPS_EP", MAX_STEPS_EP),
    ("REWARD_SCALE", REWARD_SCALE),
    ("LAMBDA_PEN",   LAMBDA_PEN),
    ("GAMMA",        GAMMA),
    ("gae_lambda",   model.gae_lambda),
    ("clip_range",   model.clip_range if not callable(model.clip_range)
                     else model.clip_range(1.0)),
    ("vf_coef",      model.vf_coef),
    ("VecNormalize pkl", os.path.basename(vn_path)),
]

df = pd.DataFrame(rows, columns=["Parameter", "Value"])
row_h, fig_h = 0.35, 0.35 * (len(df) + 1)
fig, ax = plt.subplots(figsize=(7, fig_h))
ax.axis("off")
tbl = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
    cellLoc="left"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.auto_set_column_width(col=[0, 1])
for cell in tbl.get_celld().values():
    cell.set_height(row_h / fig_h)

plt.title("Training & evaluation summary", pad=8)
plt.tight_layout()
plt.show()

# =============================================================================
# Optional: quick evaluator function (no plotting)
# =============================================================================
def run_one_episode(J_val: float = J_EVAL, wn_val: float = WN, zeta_val: float = ZETA):
    """
    Return (gamma_disc_return, gamma_disc_reward, gamma_disc_dtheta)
    for one deterministic episode with a chosen (J, wn, zeta).
    """
    def _make():
        base = HighOrderDroneEnv(
            delta_t=DELTA_T,
            J=J_val,
            wn=wn_val,
            zeta=zeta_val,
            theta_star_tau=TAU_EVAL,
            print_specs_once=False,
        )
        env = RewardCostWrapper(base, lam=LAMBDA_PEN)
        env = ReducedObsActionAdapter(env)
        env.seed(0)
        if USE_FIXED_START:
            env = FixedStart(env, FIXED_STATE)
        return env

    raw = DummyVecEnv([_make])
    ve  = VecNormalize.load(str(vn_path), venv=raw)
    ve.training = False
    ve.norm_reward = False

    mdl = PPO.load(model_path, env=ve, device="cpu")

    obs = ve.reset()
    raw_rs, costs = [], []
    for _ in range(MAX_STEPS_EP):
        act, _ = mdl.predict(obs, deterministic=True)
        obs, _, done, info = ve.step(act)
        raw_rs.append(float(info[0]["raw_r"]))
        costs.append(float(info[0]["cost"]))
        if done:
            break

    g = mdl.gamma
    pw = g ** np.arange(len(raw_rs))
    disc_r = float(pw @ np.asarray(raw_rs))
    disc_c = float(pw @ np.asarray(costs))
    disc_ret = disc_r - LAMBDA_PEN * disc_c
    return disc_ret, disc_r, disc_c
