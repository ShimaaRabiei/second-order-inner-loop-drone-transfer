# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 13:10:03 2025

@author: RABIEI
"""

# -*- coding: utf-8 -*-
"""
COMP_ALL_EARLY_SUM_600_rank_traj_plots_LAMBDA_COMPARE.py

Evaluate PPO policies trained with:
    R_early summer .... where we had reward at target .py

RunIDs (λ mapping):
    l0_2_000_000   -> lambda = 0
    l5_2_000_000   -> lambda = 5
    l10_2_000_000  -> lambda = 10
    l15_2_000_000  -> lambda = 15

Evaluation protocol (analog of your FULL-model compare script):
- Load fixed initial conditions from: _333_eval_init_states.npy   (x, xdot, z, zdot)
- Fix zeta = ZETA_EVAL
- Sweep omega_n over WN_VALUES
- For each (wn, lambda/run_id): run one rollout per init state, log:
    * position error ||(x,z)-(9,9)||
    * cumulative discounted return  Σ γ^k (raw_r - λ*cost)
    * cumulative discounted reward  Σ γ^k (raw_r)
    * θ, θ*, tracking errors (pre-step and post-step)
    * x,z,vx,vz trajectories
    * ΔT action trajectory
    * env theta_star_dot_est trajectory

"""

import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# =============================================================================
# Project paths / imports
# =============================================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "lambda_policies"
PLOT_BASE_DIR = PROJECT_ROOT / "results" / "lambda_comparison"
PLOT_BASE_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import env_second_order_inner_loop as HEARLY

# Optional cross-check with FULL gain computation
try:
    import theta_dot_reference_utils as HFULL
    _HAS_FULL = True
except Exception:
    HFULL = None
    _HAS_FULL = False


# =============================================================================
# Evaluation settings
# =============================================================================

ZETA_EVAL = 0.4

# ω_n values to sweep
WN_VALUES = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


# =============================================================================
# Run IDs and λ mapping (as you requested)
# =============================================================================
RUNS = [
    ("l0_2_000_000",  0.0,  r"$\lambda=0$"),
    ("l5_2_000_000",  5.0,  r"$\lambda=5$"),
    ("l10_2_000_000", 10.0, r"$\lambda=10$"),
    ("l15_2_000_000", 15.0, r"$\lambda=15$"),
]

# Distinct colors (4 curves)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# =============================================================================
# Load initial conditions (x, xdot, z, zdot)
# =============================================================================
INIT_STATES_PATH = DATA_DIR / "eval_initial_states_333.npy"
if not INIT_STATES_PATH.exists():
    raise FileNotFoundError(f"Initial state file not found: {INIT_STATES_PATH}")

INIT_STATES = np.load(str(INIT_STATES_PATH)).astype(np.float32)
if INIT_STATES.ndim != 2 or INIT_STATES.shape[1] < 4:
    raise ValueError(
        f"_333_eval_init_states.npy must have shape (N,4) or (N,>=4); got {INIT_STATES.shape}"
    )

N_ROLLOUTS = INIT_STATES.shape[0]  # one episode per initial state


# =============================================================================
# Helpers
# =============================================================================
def gains_from_specs(wn: float, zeta: float, J: float):
    """Same as H_THETA_DOT_STAR_ESTIMATE.py: Kp = J*wn^2, Kd = 2*J*zeta*wn."""
    return J * (wn ** 2), 2.0 * J * zeta * wn


def _find_vecnormalize_pkl(run_id: str) -> str:
    mapping = {
        "l0_2_000_000": "lambda_0_vecnormalize.pkl",
        "l5_2_000_000": "lambda_5_vecnormalize.pkl",
        "l10_2_000_000": "lambda_10_vecnormalize.pkl",
        "l15_2_000_000": "lambda_15_vecnormalize.pkl",
    }
    filename = mapping.get(run_id, f"{run_id}_vecnormalize.pkl")
    path = MODEL_DIR / filename
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"VecNormalize stats not found for run_id={run_id}: {path}")


def _find_model_zip(run_id: str) -> str:
    mapping = {
        "l0_2_000_000": "lambda_0_policy.zip",
        "l5_2_000_000": "lambda_5_policy.zip",
        "l10_2_000_000": "lambda_10_policy.zip",
        "l15_2_000_000": "lambda_15_policy.zip",
    }
    filename = mapping.get(run_id, f"{run_id}_policy.zip")
    path = MODEL_DIR / filename
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Model file not found for run_id={run_id}: {path}")


def _unwrap_to_base_env(env):
    """Unwrap a Gym env down to the base (non-Wrapper) env."""
    wr = env
    while isinstance(wr, gym.Wrapper):
        wr = wr.env
    return wr


class PreloadedStartEarly(gym.Wrapper):
    """
    For EARLY model (H_early_sum_mmer_2nd_theta.py):

    On reset:
    - call underlying reset()
    - overwrite base_env.state[0:4] with a preset IC [x, xdot, z, zdot]
    - return the *correct* 4D observation matching the overwritten state.
    """

    def __init__(self, env, init_states: np.ndarray):
        super().__init__(env)
        init_states = np.asarray(init_states, dtype=np.float32)
        if init_states.ndim != 2 or init_states.shape[1] < 4:
            raise ValueError(
                f"init_states must have shape (N,4) or (N,>=4); got {init_states.shape}"
            )
        self.init_states = init_states
        self._idx = 0

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)

        base = _unwrap_to_base_env(self.env)
        s = np.array(base.state, dtype=np.float32)

        ic = self.init_states[self._idx % len(self.init_states)]
        self._idx += 1

        s[0] = ic[0]  # x
        s[1] = ic[1]  # xdot
        s[2] = ic[2]  # z
        s[3] = ic[3]  # zdot

        base.state = s

        # IMPORTANT: EARLY agent obs is [x, xdot, z, zdot]
        obs4 = base.state[:4].astype(np.float32)
        return obs4


class ThetaTapEarly(gym.Wrapper):
    """
    Add useful signals into info at each step:
        info['x'], info['z'], info['vx'], info['vz']
        info['theta_actual']      -- θ (post-step) from base state index 4
        info['theta_dot']         -- θ̇ (post-step) from base state index 5
        info['theta_star']        -- θ★ (from action[1])
        info['theta_star_dot_est']-- base_env.theta_star_dot_est
    """

    def step(self, action):
        obs, r, done, info = self.env.step(action)

        base = _unwrap_to_base_env(self.env)
        s = np.array(base.state, dtype=float)

        a_flat = np.asarray(action, dtype=float).reshape(-1)
        theta_star = float(a_flat[1]) if a_flat.size >= 2 else 0.0

        info = dict(info)
        info.update(
            {
                "x": float(s[0]),
                "vx": float(s[1]),
                "z": float(s[2]),
                "vz": float(s[3]),
                "theta_actual": float(s[4]) if s.size > 4 else 0.0,
                "theta_dot": float(s[5]) if s.size > 5 else 0.0,
                "theta_star": theta_star,
                "theta_star_dot_est": float(getattr(base, "theta_star_dot_est", 0.0)),
            }
        )
        return obs, r, done, info


def _build_eval_env_early(run_id: str, wn_val: float, zeta_val: float, lam_val: float, init_states: np.ndarray):
    """
    Build VecNormalize-wrapped evaluation environment for EARLY model,
    with specified (wn, zeta) and specified lambda (via RewardCostWrapper).
    Uses preloaded initial states.
    """
    # Cross-check gains against FULL script (optional)
    J = float(getattr(HEARLY, "INERTIA_J", 0.02))
    kp, kd = gains_from_specs(wn_val, zeta_val, J)
    if _HAS_FULL and hasattr(HFULL, "INERTIA_J"):
        J_full = float(HFULL.INERTIA_J)
        kp_full, kd_full = HFULL.gains_from_specs(wn_val, zeta_val, J_full)
        if abs(J - J_full) > 1e-12 or abs(kp - kp_full) > 1e-9 or abs(kd - kd_full) > 1e-9:
            print("[WARN] Gain mismatch vs FULL code:")
            print(f"  EARLY: J={J}, Kp={kp}, Kd={kd}")
            print(f"  FULL : J={J_full}, Kp={kp_full}, Kd={kd_full}")

    def _make():
        env = HEARLY.HighOrderDroneEnv(
            delta_t=float(getattr(HEARLY, "DELTA_T", 0.05)),
            J=J,
            wn=float(wn_val),
            zeta=float(zeta_val),
            print_specs_once=True,  # print once per env construction (fine in eval)
        )
        env = HEARLY.RewardCostWrapper(env, lam=float(lam_val))
        env = HEARLY.ReducedObsActionAdapter(env)
        env = PreloadedStartEarly(env, init_states)
        env = ThetaTapEarly(env)
        return env

    raw = DummyVecEnv([_make])

    vec_path = _find_vecnormalize_pkl(run_id)
    vec = VecNormalize.load(vec_path, venv=raw)
    vec.training = False
    vec.norm_reward = False
    return vec


# =============================================================================
# Rollout + metrics
# =============================================================================
def run_rollouts_metrics_early(run_id: str, lam_val: float, wn_val: float, zeta_val: float, init_states: np.ndarray):
    """
    Runs N_ROLLOUTS episodes (one per initial state) for EARLY model.

    Logs (per episode) then aggregates mean/std:
      - position error ||(x,z)-(9,9)||
      - cumulative discounted return (raw_r - λ*cost)
      - cumulative discounted reward (raw_r)
      - cumulative discounted theta (post-step θ_{k+1})
      - cumulative discounted theta_star (θ★_k)
      - cumulative discounted tracking errors:
            post-step: |θ_{k+1} - θ★_k|
            pre-step : |θ_k - θ★_k|
      - trajectories: x,z,vx,vz, theta_pre, theta_star, delta_thrust, theta_star_dot_est
    """
    eval_env = _build_eval_env_early(run_id, wn_val, zeta_val, lam_val, init_states)
    model_path = _find_model_zip(run_id)
    model = PPO.load(model_path, env=eval_env, device="cpu")

    # Determine dt and max steps from base env
    inner = eval_env.venv.envs[0]
    base = _unwrap_to_base_env(inner)
    dt_used = float(getattr(base, "delta_t", getattr(HEARLY, "DELTA_T", 0.05)))
    max_steps_ep = int(getattr(base, "max_steps", getattr(HEARLY, "MAX_STEPS_EP", 400)))

    gamma_local = float(getattr(model, "gamma", getattr(HEARLY, "GAMMA", 0.994)))

    to_flat = lambda x: np.asarray(x).reshape(-1)

    # Episode lists
    ts_list = []
    pos_err_list = []
    cum_disc_ret_list = []
    cum_disc_rew_list = []

    cum_disc_theta_list = []
    cum_disc_theta_star_list = []
    cum_disc_tracking_post_list = []
    cum_disc_tracking_pre_list = []

    # trajectories
    x_traj_list = []
    z_traj_list = []
    vx_traj_list = []
    vz_traj_list = []
    theta_pre_traj_list = []
    theta_star_traj_list = []
    delta_thrust_traj_list = []
    theta_star_dot_traj_list = []

    print(f"\n[RUN CONFIG] EARLY model: run_id={run_id}, lambda={lam_val}, wn={wn_val}, zeta={zeta_val}")

    for ep in range(N_ROLLOUTS):
        print(f"  Episode {ep+1}/{N_ROLLOUTS} for run_id={run_id}, wn={wn_val}")
        obs = eval_env.reset()

        # Initial (unnormalized) obs: [x, xdot, z, zdot]
        real0 = to_flat(eval_env.unnormalize_obs(obs))
        x_hist = [float(real0[0])]
        vx_hist = [float(real0[1])]
        z_hist = [float(real0[2])]
        vz_hist = [float(real0[3])]
        ts_pos = [0.0]

        step_returns = []
        raw_rewards = []

        theta_pre_seq = []
        theta_post_seq = []
        theta_star_seq = []
        delta_thrust_seq = []
        theta_star_dot_seq = []

        # get handle to base env for theta_pre reading
        inner0 = eval_env.venv.envs[0]
        base0 = _unwrap_to_base_env(inner0)

        for k in range(max_steps_ep):
            # theta_k (pre-step) from base state
            s_pre = np.array(base0.state, dtype=float)
            theta_pre_now = float(s_pre[4]) if s_pre.size > 4 else 0.0
            theta_pre_seq.append(theta_pre_now)

            # Action from policy
            action, _ = model.predict(obs, deterministic=True)
            a_flat = np.asarray(action, dtype=float).reshape(-1)
            delta_thrust_now = float(a_flat[0]) if a_flat.size >= 1 else 0.0
            theta_star_now = float(a_flat[1]) if a_flat.size >= 2 else 0.0

            delta_thrust_seq.append(delta_thrust_now)
            theta_star_seq.append(theta_star_now)

            # Step
            obs, r, done, info = eval_env.step(action)

            info0 = info[0] if isinstance(info, (list, tuple)) else info

            # post-step kinematics from info 
            x_hist.append(float(info0["x"]))
            vx_hist.append(float(info0["vx"]))
            z_hist.append(float(info0["z"]))
            vz_hist.append(float(info0["vz"]))
            ts_pos.append((k + 1) * dt_used)

            # post-step theta
            theta_post_now = float(info0.get("theta_actual", 0.0))
            theta_post_seq.append(theta_post_now)

            # env-estimated theta_star_dot
            theta_star_dot_seq.append(float(info0.get("theta_star_dot_est", 0.0)))

            # reward/cost
            r_scalar = float(to_flat(r)[0])
            raw_r = float(info0.get("raw_r", r_scalar))
            cost_val = float(info0.get("cost", 0.0))

            raw_rewards.append(raw_r)
            step_returns.append(raw_r - lam_val * cost_val)

            if done:
                break

        # arrays
        x_hist = np.asarray(x_hist, dtype=float)
        vx_hist = np.asarray(vx_hist, dtype=float)
        z_hist = np.asarray(z_hist, dtype=float)
        vz_hist = np.asarray(vz_hist, dtype=float)
        ts_pos = np.asarray(ts_pos, dtype=float)

        pos_err = np.sqrt((x_hist - 9.0) ** 2 + (z_hist - 9.0) ** 2)

        step_returns_arr = np.asarray(step_returns, dtype=float)
        raw_rewards_arr = np.asarray(raw_rewards, dtype=float)

        powers = gamma_local ** np.arange(len(step_returns_arr), dtype=float)
        disc_ret_terms = powers * step_returns_arr
        disc_rew_terms = powers * raw_rewards_arr

        cum_disc_ret = np.concatenate([[0.0], np.cumsum(disc_ret_terms)])
        cum_disc_rew = np.concatenate([[0.0], np.cumsum(disc_rew_terms)])

        ts_list.append(ts_pos)
        pos_err_list.append(pos_err)
        cum_disc_ret_list.append(cum_disc_ret)
        cum_disc_rew_list.append(cum_disc_rew)

        # trajectories
        x_traj_list.append(x_hist)
        z_traj_list.append(z_hist)
        vx_traj_list.append(vx_hist)
        vz_traj_list.append(vz_hist)

        theta_pre_seq = np.asarray(theta_pre_seq, dtype=float)
        theta_post_seq = np.asarray(theta_post_seq, dtype=float)
        theta_star_seq = np.asarray(theta_star_seq, dtype=float)
        delta_thrust_seq = np.asarray(delta_thrust_seq, dtype=float)
        theta_star_dot_seq = np.asarray(theta_star_dot_seq, dtype=float)

        theta_pre_traj_list.append(theta_pre_seq)
        theta_star_traj_list.append(theta_star_seq)
        delta_thrust_traj_list.append(delta_thrust_seq)
        theta_star_dot_traj_list.append(theta_star_dot_seq)

        # discounted theta (post), theta_star, tracking
        if theta_post_seq.size > 0:
            powers_theta = gamma_local ** np.arange(theta_post_seq.size, dtype=float)

            disc_theta_terms = powers_theta * theta_post_seq
            disc_theta_star_terms = powers_theta * theta_star_seq

            tracking_post_abs = np.abs(theta_post_seq - theta_star_seq)  # |θ_{k+1} - θ*_k|
            tracking_pre_abs = np.abs(theta_pre_seq - theta_star_seq)    # |θ_k - θ*_k|

            disc_tracking_post_terms = powers_theta * tracking_post_abs
            disc_tracking_pre_terms = powers_theta * tracking_pre_abs

            cum_theta = np.concatenate([[0.0], np.cumsum(disc_theta_terms)])
            cum_theta_star = np.concatenate([[0.0], np.cumsum(disc_theta_star_terms)])
            cum_tracking_post = np.concatenate([[0.0], np.cumsum(disc_tracking_post_terms)])
            cum_tracking_pre = np.concatenate([[0.0], np.cumsum(disc_tracking_pre_terms)])
        else:
            cum_theta = np.array([0.0], dtype=float)
            cum_theta_star = np.array([0.0], dtype=float)
            cum_tracking_post = np.array([0.0], dtype=float)
            cum_tracking_pre = np.array([0.0], dtype=float)

        cum_disc_theta_list.append(cum_theta)
        cum_disc_theta_star_list.append(cum_theta_star)
        cum_disc_tracking_post_list.append(cum_tracking_post)
        cum_disc_tracking_pre_list.append(cum_tracking_pre)

    # Aggregate across episodes with padding
    max_len = max(len(ts) for ts in ts_list)
    ts_grid = dt_used * np.arange(max_len, dtype=float)

    def _pad_to_max(arr_list):
        n = len(arr_list)
        out = np.full((n, max_len), np.nan, dtype=float)
        for i, a in enumerate(arr_list):
            L = len(a)
            out[i, :L] = a
        return out

    pos_err_runs = _pad_to_max(pos_err_list)
    ret_runs = _pad_to_max(cum_disc_ret_list)
    rew_runs = _pad_to_max(cum_disc_rew_list)
    theta_runs = _pad_to_max(cum_disc_theta_list)
    theta_star_runs = _pad_to_max(cum_disc_theta_star_list)
    tracking_post_runs = _pad_to_max(cum_disc_tracking_post_list)
    tracking_pre_runs = _pad_to_max(cum_disc_tracking_pre_list)

    x_runs = _pad_to_max(x_traj_list)
    z_runs = _pad_to_max(z_traj_list)
    vx_runs = _pad_to_max(vx_traj_list)
    vz_runs = _pad_to_max(vz_traj_list)
    theta_pre_runs = _pad_to_max(theta_pre_traj_list)
    theta_star_traj_runs = _pad_to_max(theta_star_traj_list)
    delta_thrust_runs = _pad_to_max(delta_thrust_traj_list)
    theta_star_dot_runs = _pad_to_max(theta_star_dot_traj_list)

    metrics = {
        "ts": ts_grid,
        "pos_err_mean": np.nanmean(pos_err_runs, axis=0),
        "pos_err_std": np.nanstd(pos_err_runs, axis=0),
        "cum_ret_mean": np.nanmean(ret_runs, axis=0),
        "cum_ret_std": np.nanstd(ret_runs, axis=0),
        "cum_rew_mean": np.nanmean(rew_runs, axis=0),
        "cum_rew_std": np.nanstd(rew_runs, axis=0),

        "cum_theta_mean": np.nanmean(theta_runs, axis=0),
        "cum_theta_std": np.nanstd(theta_runs, axis=0),
        "cum_theta_star_mean": np.nanmean(theta_star_runs, axis=0),
        "cum_theta_star_std": np.nanstd(theta_star_runs, axis=0),

        "cum_tracking_post_mean": np.nanmean(tracking_post_runs, axis=0),
        "cum_tracking_post_std": np.nanstd(tracking_post_runs, axis=0),
        "cum_tracking_pre_mean": np.nanmean(tracking_pre_runs, axis=0),
        "cum_tracking_pre_std": np.nanstd(tracking_pre_runs, axis=0),

        "x_mean": np.nanmean(x_runs, axis=0),
        "x_std": np.nanstd(x_runs, axis=0),
        "z_mean": np.nanmean(z_runs, axis=0),
        "z_std": np.nanstd(z_runs, axis=0),
        "vx_mean": np.nanmean(vx_runs, axis=0),
        "vx_std": np.nanstd(vx_runs, axis=0),
        "vz_mean": np.nanmean(vz_runs, axis=0),
        "vz_std": np.nanstd(vz_runs, axis=0),

        "theta_pre_mean": np.nanmean(theta_pre_runs, axis=0),
        "theta_pre_std": np.nanstd(theta_pre_runs, axis=0),

        "theta_star_traj_mean": np.nanmean(theta_star_traj_runs, axis=0),
        "theta_star_traj_std": np.nanstd(theta_star_traj_runs, axis=0),

        "delta_thrust_mean": np.nanmean(delta_thrust_runs, axis=0),
        "delta_thrust_std": np.nanstd(delta_thrust_runs, axis=0),

        "theta_star_dot_mean": np.nanmean(theta_star_dot_runs, axis=0),
        "theta_star_dot_std": np.nanstd(theta_star_dot_runs, axis=0),
    }
    return metrics


# =============================================================================
# Per-ω_n evaluation (all λ runs) + plots
# =============================================================================
def evaluate_for_wn(wn_val: float, out_dir: str, zeta_eval: float = ZETA_EVAL):
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for (run_id, lam_val, lam_label), color in zip(RUNS, COLORS):
        metrics = run_rollouts_metrics_early(
            run_id=run_id,
            lam_val=lam_val,
            wn_val=wn_val,
            zeta_val=zeta_eval,
            init_states=INIT_STATES,
        )
        res = {
            "run_id": run_id,
            "lambda": lam_val,
            "lambda_label": lam_label,
            "label": rf"EARLY (H_early), {lam_label}, $\omega_n={wn_val:g}$, $\zeta={zeta_eval:g}$",
            "color": color,
        }
        res.update(metrics)
        results.append(res)

    title_suffix = (
        r" --- EARLY model (H_early_sum_mmer_2nd_theta); "
        r"initial states from $\_333\_eval\_init\_states.npy$, "
        rf"evaluated at $(\omega_n,\zeta)=({wn_val:g},{zeta_eval:g})$"
    )

    # baseline index (lambda=0)
    baseline_idx = None
    for i, res in enumerate(results):
        if abs(res["lambda"]) < 1e-12:
            baseline_idx = i
            break

    # rank by final cumulative discounted reward
    ranking = []
    for res in results:
        cum_rew = res["cum_rew_mean"]
        valid = ~np.isnan(cum_rew)
        final_val = float(cum_rew[valid][-1]) if np.any(valid) else float("nan")
        ranking.append((final_val, res["lambda"], res["run_id"]))
    ranking.sort(key=lambda t: t[0], reverse=True)

    print(f"\n=== Ranking by final cumulative discounted reward (EARLY, wn={wn_val}) ===")
    for rnk, (final_val, lam_val, run_id) in enumerate(ranking, start=1):
        print(f"Rank {rnk:2d}: lambda={lam_val:6.2f}, final discounted reward={final_val:+.6f}, run_id={run_id}")

    with open(os.path.join(out_dir, f"ranking_cum_rew_wn_{wn_val:g}.txt"), "w") as f:
        f.write(f"Ranking by final cumulative discounted reward (EARLY, wn={wn_val})\n")
        for rnk, (final_val, lam_val, run_id) in enumerate(ranking, start=1):
            f.write(f"Rank {rnk:2d}: lambda={lam_val:6.2f}, final discounted reward={final_val:+.6f}, run_id={run_id}\n")

    # ----------------------------
    # Plot 00: mean x-z trajectory
    # ----------------------------
    fig = plt.figure(figsize=(7, 7))
    for res in results:
        plt.plot(res["x_mean"], res["z_mean"], color=res["color"], label=res["lambda_label"])
    plt.scatter([9.0], [9.0], marker="x", s=80, color="k", label="target (9,9)")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Mean trajectory in x-z plane" + title_suffix)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "00_mean_xz_trajectory.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 01: position error vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["pos_err_mean"], res["pos_err_std"]
        plt.plot(ts, m, color=res["color"], label=res["label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"position error $\|(x,z)-(9,9)\|$")
    plt.title("Position error vs time" + title_suffix)
    plt.axhline(0.0, linewidth=1, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_pos_err_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 02: cumulative discounted return vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["cum_ret_mean"], res["cum_ret_std"]
        plt.plot(ts, m, color=res["color"], label=res["label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel("cumulative discounted return")
    plt.title("Cumulative discounted return vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_cum_return_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 03: cumulative discounted reward vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["cum_rew_mean"], res["cum_rew_std"]
        plt.plot(ts, m, color=res["color"], label=res["label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel("cumulative discounted reward")
    plt.title("Cumulative discounted reward vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_cum_reward_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 04: cumulative discounted θ and θ★ vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        plt.plot(ts, res["cum_theta_mean"], color=res["color"], linestyle="-",
                 label=res["lambda_label"] + r" (actual $\theta$)")
        plt.plot(ts, res["cum_theta_star_mean"], color=res["color"], linestyle="--",
                 label=res["lambda_label"] + r" ($\theta^\star$)")
    plt.xlabel("time [s]")
    plt.ylabel(r"cumulative discounted $\theta$ / $\theta^\star$")
    plt.title(r"Cumulative discounted $\theta$ and $\theta^\star$ vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_cum_theta_and_theta_star_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 05: cumulative discounted tracking error (POST-step)
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["cum_tracking_post_mean"], res["cum_tracking_post_std"]
        plt.plot(ts, m, color=res["color"], label=res["lambda_label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"cumulative discounted $|\theta_{k+1} - \theta^\star_k|$")
    plt.title(r"Cumulative discounted tracking error (post-step) vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_cum_tracking_post_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 06: cumulative discounted tracking error (PRE-step)
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["cum_tracking_pre_mean"], res["cum_tracking_pre_std"]
        plt.plot(ts, m, color=res["color"], label=res["lambda_label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"cumulative discounted $|\theta_k - \theta^\star_k|$")
    plt.title(r"Cumulative discounted tracking error (pre-step) vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_cum_tracking_pre_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 07: x position error (x - 9) vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        x_err = res["x_mean"] - 9.0
        s = res["x_std"]
        plt.plot(ts, x_err, color=res["color"], label=res["lambda_label"])
        plt.fill_between(ts, x_err - s, x_err + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"$x - 9$")
    plt.title("X position error vs time" + title_suffix)
    plt.axhline(0.0, linewidth=1, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_x_position_error_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 08: z position error (z - 9) vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        z_err = res["z_mean"] - 9.0
        s = res["z_std"]
        plt.plot(ts, z_err, color=res["color"], label=res["lambda_label"])
        plt.fill_between(ts, z_err - s, z_err + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"$z - 9$")
    plt.title("Z position error vs time" + title_suffix)
    plt.axhline(0.0, linewidth=1, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_z_position_error_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 09: vx vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["vx_mean"], res["vx_std"]
        plt.plot(ts, m, color=res["color"], label=res["lambda_label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"$v_x$")
    plt.title("X velocity vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_vx_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 10: vz vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["vz_mean"], res["vz_std"]
        plt.plot(ts, m, color=res["color"], label=res["lambda_label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"$v_z$")
    plt.title("Z velocity vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_vz_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 11: θ_k (pre-step) and θ★_k vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        theta_m, theta_s = res["theta_pre_mean"], res["theta_pre_std"]
        theta_star_m = res["theta_star_traj_mean"]

        plt.plot(ts, theta_m, color=res["color"], linestyle="-",
                 label=res["lambda_label"] + r" ($\theta_k$)")
        plt.fill_between(ts, theta_m - theta_s, theta_m + theta_s,
                         color=res["color"], alpha=0.1, linewidth=0)

        plt.plot(ts, theta_star_m, color=res["color"], linestyle="--",
                 label=res["lambda_label"] + r" ($\theta^\star_k$)")
    plt.xlabel("time [s]")
    plt.ylabel(r"$\theta_k$ and $\theta^\star_k$")
    plt.title(r"Pre-step $\theta_k$ and reference $\theta^\star_k$ vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "11_theta_and_theta_star_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 12: ΔT_k vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["delta_thrust_mean"], res["delta_thrust_std"]
        plt.plot(ts, m, color=res["color"], label=res["lambda_label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"$\Delta T_k$")
    plt.title(r"Delta thrust $\Delta T_k$ vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "12_delta_thrust_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 13: θ̇★_est,k vs time
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    for res in results:
        ts = res["ts"]
        m, s = res["theta_star_dot_mean"], res["theta_star_dot_std"]
        plt.plot(ts, m, color=res["color"], label=res["lambda_label"])
        plt.fill_between(ts, m - s, m + s, color=res["color"], alpha=0.2, linewidth=0)
    plt.xlabel("time [s]")
    plt.ylabel(r"$\dot{\theta}^\star_{\mathrm{est},k}$")
    plt.title(r"Env-estimated $\dot{\theta}^\star_k$ vs time" + title_suffix)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "13_theta_star_dot_est_vs_time.png"), dpi=300)
    plt.show()

    # ----------------------------------------------------------
    # Plot 14: (cum reward) differences vs baseline λ=0 over time
    #         (lam15-lam0, lam10-lam0, lam5-lam0)
    # ----------------------------------------------------------
    if baseline_idx is not None:
        base_rew = results[baseline_idx]["cum_rew_mean"]
        fig = plt.figure(figsize=(10, 6))
        for i, res in enumerate(results):
            if i == baseline_idx:
                continue
            diff = res["cum_rew_mean"] - base_rew
            plt.plot(res["ts"], diff, color=res["color"],
                     label=res["lambda_label"] + r" $-$ " + results[baseline_idx]["lambda_label"])
        plt.axhline(0.0, linewidth=1, alpha=0.8, linestyle="--", color="k")
        plt.xlabel("time [s]")
        plt.ylabel(r"$J_{\mathrm{rew}}(\lambda) - J_{\mathrm{rew}}(\lambda=0)$ (cumulative, discounted)")
        plt.title("Cumulative discounted reward difference vs baseline" + title_suffix)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "14_cum_reward_diff_vs_lambda0_vs_time.png"), dpi=300)
        plt.show()

        # ----------------------------------------------------------
        # Plot 15: (cum return) differences vs baseline λ=0 over time
        # ----------------------------------------------------------
        base_ret = results[baseline_idx]["cum_ret_mean"]
        fig = plt.figure(figsize=(10, 6))
        for i, res in enumerate(results):
            if i == baseline_idx:
                continue
            diff = res["cum_ret_mean"] - base_ret
            plt.plot(res["ts"], diff, color=res["color"],
                     label=res["lambda_label"] + r" $-$ " + results[baseline_idx]["lambda_label"])
        plt.axhline(0.0, linewidth=1, alpha=0.8, linestyle="--", color="k")
        plt.xlabel("time [s]")
        plt.ylabel(r"$J_{\mathrm{return}}(\lambda) - J_{\mathrm{return}}(\lambda=0)$ (cumulative, discounted)")
        plt.title("Cumulative discounted return difference vs baseline" + title_suffix)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "15_cum_return_diff_vs_lambda0_vs_time.png"), dpi=300)
        plt.show()

    # final cumulative discounted reward (for summary across wn)
    final_cum_rew = []
    for res in results:
        cum_rew = res["cum_rew_mean"]
        valid = ~np.isnan(cum_rew)
        final_val = float(cum_rew[valid][-1]) if np.any(valid) else float("nan")
        final_cum_rew.append(final_val)
    final_cum_rew = np.asarray(final_cum_rew, dtype=float)

    return results, final_cum_rew


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    num_wn = len(WN_VALUES)
    num_runs = len(RUNS)

    all_final_cum_rew = np.full((num_wn, num_runs), np.nan, dtype=float)

    # baseline index in RUNS (lambda=0)
    baseline_idx = None
    for j, (_, lam, _) in enumerate(RUNS):
        if abs(lam) < 1e-12:
            baseline_idx = j
            break
    if baseline_idx is None:
        print("Warning: no λ=0 run found; summary improvement plot vs λ=0 will be skipped.")

    # Evaluate for each wn
    for i, wn_val in enumerate(WN_VALUES):
        wn_dir_name = f"wn_{wn_val:g}_{timestamp_str}"
        wn_dir = os.path.join(PLOT_BASE_DIR, wn_dir_name)
        print(f"\n===== Evaluating EARLY model for ω_n = {wn_val} (folder: {wn_dir_name}) =====")
        _, final_cum_rew = evaluate_for_wn(
            wn_val=wn_val,
            out_dir=wn_dir,
            zeta_eval=ZETA_EVAL,
        )
        all_final_cum_rew[i, :] = final_cum_rew

    # Summary across wn: final_cum_rew(λ, wn) - final_cum_rew(λ=0, wn) vs wn
    summary_dir = os.path.join(PLOT_BASE_DIR, f"summary_{timestamp_str}")
    os.makedirs(summary_dir, exist_ok=True)

    if baseline_idx is not None:
        baseline = all_final_cum_rew[:, baseline_idx][:, None]
        diffs = all_final_cum_rew - baseline

        fig = plt.figure(figsize=(10, 6))
        for j, ((run_id, lam_val, lam_label), color) in enumerate(zip(RUNS, COLORS)):
            if j == baseline_idx:
                plt.plot(WN_VALUES, diffs[:, j], linestyle=":", marker="o",
                         color=color, label=lam_label + r" (baseline, diff=0)")
                continue
            plt.plot(WN_VALUES, diffs[:, j], linestyle="-", marker="o",
                     color=color, label=lam_label + r" $-$ $\lambda=0$")
        plt.axhline(0.0, linewidth=1, alpha=0.8, linestyle="--", color="k")
        plt.xlabel(r"$\omega_n$")
        plt.ylabel("Δ final cumulative discounted reward\n" r"( $J(\lambda)$ $-$ $J(\lambda=0)$ )")
        plt.title(
            "EARLY model: improvement in final cumulative discounted reward\n"
            rf"relative to $\lambda=0$ for different $\omega_n$, $\zeta={ZETA_EVAL:g}$"
        )
        plt.legend()
        plt.tight_layout()

        summary_plot_path = os.path.join(
            summary_dir, f"final_cum_disc_reward_diff_vs_wn_zeta_{ZETA_EVAL:g}.png"
        )
        fig.savefig(summary_plot_path, dpi=300)
        plt.show()

        # Save CSV
        header_cols = ["wn"] + [f"J_lambda_{RUNS[j][1]:.3f}_minus_lambda_0" for j in range(num_runs)]
        data = np.column_stack([WN_VALUES, diffs])
        # csv_path = os.path.join(summary_dir, "final_cum_disc_reward_diff_vs_wn.csv")
        # np.savetxt(data, csv_path, delimiter=",", header=",".join(header_cols), comments="")
        csv_path = os.path.join(summary_dir, "final_cum_disc_reward_diff_vs_wn.csv")
        np.savetxt(csv_path, data, delimiter=",", header=",".join(header_cols), comments="")


        print(f"\nSummary plot saved to: {summary_plot_path}\nSummary CSV saved to:  {csv_path}")
    else:
        print("\nNo λ=0 baseline found; skipped summary difference plot and CSV.")
