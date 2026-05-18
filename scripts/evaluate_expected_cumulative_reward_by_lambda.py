# -*- coding: utf-8 -*-
"""
Plot expected cumulative reward for several lambda policies under full-model
deployment, with the reduced-model lambda=0 value shown as a baseline.

This script makes one curve plot for each zeta value:

    x-axis: omega_n
    y-axis: expected cumulative reward
    curves: full-model deployment for lambda = 0, 0.5, 3, 5 by default
    baseline: reduced-model lambda=0 expected cumulative reward

The plotted quantity is

    J_hat = mean_i sum_t gamma^t r_{i,t}

where r_{i,t} is the reward. The theta-variation cost is not included in these
plots.


Full run for zeta=0.7:
    python scripts/evaluate_expected_cumulative_reward_by_lambda.py --zeta_values 0.7 --wn_values 2 4 6 8 10 12 --lambda_values 0 0.5 3 5
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


 
# Project paths / imports
 
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_lambda_comparison as comp  # noqa: E402
import reduced_order_drone_env as RENV  # noqa: E402


 
# Small helpers
 
def lambda_to_run_id(lam_val):
    """Return the saved-model run id used by the existing lambda scripts."""
    return f"lambda_{float(lam_val):g}"


def safe_float_name(val):
    """Convert a float to a filename-safe string."""
    return f"{float(val):g}".replace("-", "m").replace(".", "p")


def unique_floats(values):
    """Remove duplicate float values while preserving input order."""
    unique = []
    for value in values:
        value = float(value)
        if not any(abs(value - old) < 1e-12 for old in unique):
            unique.append(value)
    return unique


def discounted_sum(reward_seq, gamma):
    """Compute one rollout cumulative discounted reward."""
    rewards = np.asarray(reward_seq, dtype=float).reshape(-1)
    if rewards.size == 0:
        return np.nan
    powers = float(gamma) ** np.arange(rewards.size, dtype=float)
    return float(np.dot(powers, rewards))


def mean_std(values):
    """Return mean and population standard deviation."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def get_info0(info):
    """Return the first info dictionary from a VecEnv-style info object."""
    if isinstance(info, (list, tuple)):
        return info[0]
    return info


def get_done_bool(done):
    """Return the scalar done flag from a VecEnv-style done object."""
    return bool(np.asarray(done).reshape(-1)[0])


def vec_reward_to_float(vec_reward):
    """Return a scalar reward from a VecEnv-style reward object."""
    return float(np.asarray(vec_reward, dtype=float).reshape(-1)[0])


def close_env_safely(env):
    """Close a Gym/VecEnv object without interrupting result saving."""
    try:
        env.close()
    except Exception:
        pass


def lambda_label(lam_val):
    """Return a clean legend label for a lambda value."""
    return rf"$\lambda={float(lam_val):g}$"


 
# Reduced-order fixed-start wrapper
 
def unwrap_to_base_env(env):
    """Unwrap a Gym environment down to the base environment."""
    current = env
    while isinstance(current, gym.Wrapper):
        current = current.env
    return current


class PreloadedStartReduced(gym.Wrapper):
    """
    Reset wrapper for the reduced-order model.

    On each reset, the wrapper overwrites the reduced-order state with the next
    fixed initial condition. This makes the reduced lambda=0 baseline use the
    same initial states as the full-model deployment evaluations.
    """

    def __init__(self, env, init_states):
        super().__init__(env)
        init_states = np.asarray(init_states, dtype=np.float32)
        if init_states.ndim != 2 or init_states.shape[1] < 4:
            raise ValueError(
                f"init_states must have shape (N,4) or (N,>=4); got {init_states.shape}"
            )
        self.init_states = init_states[:, :4].copy()
        self._idx = 0

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)

        base = unwrap_to_base_env(self.env)
        ic = self.init_states[self._idx % len(self.init_states)]
        self._idx += 1

        base.state = np.asarray(ic, dtype=np.float32).copy()
        return base.state.astype(np.float32)


 
# Environment builders
 
def build_reduced_eval_env(run_id, lam_val, init_states):
    """Build a VecNormalize-wrapped reduced-order evaluation environment."""

    def make_one_env():
        env = RENV.make_env(0, lam=float(lam_val))()
        env = PreloadedStartReduced(env, init_states)
        return env

    raw_env = comp.DummyVecEnv([make_one_env])
    vec_path = comp._find_vecnormalize_pkl(run_id)
    vec_env = comp.VecNormalize.load(vec_path, venv=raw_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def build_full_eval_env(run_id, lam_val, wn_val, zeta_val, init_states):
    """Build a VecNormalize-wrapped full/high-order deployment environment."""
    return comp._build_eval_env_early(
        run_id=run_id,
        wn_val=float(wn_val),
        zeta_val=float(zeta_val),
        lam_val=float(lam_val),
        init_states=init_states,
    )


 
# Reward extraction and rollout evaluation
 
def extract_reward(info0, vec_reward, require_info_reward=True):
    """
    Extract the reward used for the plot.

    For lambda > 0, the environment step reward may contain the theta-variation
    cost. Therefore, by default this function reads the reward from info["raw_r"]
    or similar keys. Use --allow_reward_fallback only if the environment reward
    is known to be the reward without the theta-variation cost.
    """
    if isinstance(info0, dict):
        for key in ("raw_r", "raw_reward", "reward_raw", "task_reward"):
            if key in info0:
                return float(info0[key])

    if require_info_reward:
        available_keys = sorted(info0.keys()) if isinstance(info0, dict) else []
        raise KeyError(
            "Could not find reward in info. Expected one of: raw_r, raw_reward, "
            "reward_raw, task_reward. "
            f"Available info keys: {available_keys}. "
            "Use --allow_reward_fallback only if the environment reward is known "
            "to exclude the theta-variation cost."
        )

    return vec_reward_to_float(vec_reward)


def evaluate_full_expected_cumulative_reward(
    run_id,
    lam_val,
    wn_val,
    zeta_val,
    init_states,
    require_info_reward=True,
    episode_log_every=25,
):
    """Evaluate one lambda policy deployed on the full model for one gain setting."""
    print("\n===================================================")
    print("Full-model deployment evaluation")
    print(f"run_id   = {run_id}")
    print(f"lambda   = {lam_val:g}")
    print(f"omega_n  = {wn_val:g}")
    print(f"zeta     = {zeta_val:g}")
    print(f"rollouts = {len(init_states)}")
    print("quantity = expected cumulative reward")
    print("===================================================")

    eval_env = build_full_eval_env(run_id, lam_val, wn_val, zeta_val, init_states)
    try:
        model_path = comp._find_model_zip(run_id)
        model = comp.PPO.load(model_path, env=eval_env, device="cpu")

        inner = eval_env.venv.envs[0]
        base = comp._unwrap_to_base_env(inner)
        max_steps_ep = int(
            getattr(
                base,
                "max_steps",
                getattr(comp.HEARLY, "MAX_STEPS_EP", getattr(RENV, "MAX_STEPS_EP", 400)),
            )
        )
        gamma = float(getattr(model, "gamma", getattr(RENV, "GAMMA", 0.994)))

        cumulative_rewards = []
        lengths = []

        for ep in range(int(len(init_states))):
            if episode_log_every and (
                ep == 0
                or ep + 1 == len(init_states)
                or (ep + 1) % int(episode_log_every) == 0
            ):
                print(f"  Episode {ep + 1}/{len(init_states)}")

            obs = eval_env.reset()
            rewards = []

            for _ in range(max_steps_ep):
                action, _ = model.predict(obs, deterministic=True)
                obs, vec_reward, done, info = eval_env.step(action)

                info0 = get_info0(info)
                reward = extract_reward(
                    info0=info0,
                    vec_reward=vec_reward,
                    require_info_reward=require_info_reward,
                )
                rewards.append(reward)

                if get_done_bool(done):
                    break

            cumulative_rewards.append(discounted_sum(rewards, gamma))
            lengths.append(len(rewards))

    finally:
        close_env_safely(eval_env)

    return (
        np.asarray(cumulative_rewards, dtype=float),
        np.asarray(lengths, dtype=int),
        gamma,
    )


def evaluate_reduced_lambda0_baseline(
    init_states,
    episode_log_every=25,
    allow_reward_fallback=True,
):
    """Evaluate the lambda=0 policy on the reduced-order model once."""
    lam_val = 0.0
    run_id = lambda_to_run_id(lam_val)

    print("\n===================================================")
    print("Reduced-model lambda=0 baseline evaluation")
    print(f"run_id   = {run_id}")
    print("lambda   = 0")
    print(f"rollouts = {len(init_states)}")
    print("quantity = expected cumulative reward")
    print("note     = independent of omega_n and zeta")
    print("===================================================")

    eval_env = build_reduced_eval_env(run_id=run_id, lam_val=lam_val, init_states=init_states)
    try:
        model_path = comp._find_model_zip(run_id)
        model = comp.PPO.load(model_path, env=eval_env, device="cpu")

        inner = eval_env.venv.envs[0]
        base = comp._unwrap_to_base_env(inner)
        max_steps_ep = int(getattr(base, "max_steps", getattr(RENV, "MAX_STEPS_EP", 400)))
        gamma = float(getattr(model, "gamma", getattr(RENV, "GAMMA", 0.994)))

        cumulative_rewards = []
        lengths = []

        for ep in range(int(len(init_states))):
            if episode_log_every and (
                ep == 0
                or ep + 1 == len(init_states)
                or (ep + 1) % int(episode_log_every) == 0
            ):
                print(f"  Episode {ep + 1}/{len(init_states)}")

            obs = eval_env.reset()
            rewards = []

            for _ in range(max_steps_ep):
                action, _ = model.predict(obs, deterministic=True)
                obs, vec_reward, done, info = eval_env.step(action)

                info0 = get_info0(info)
                reward = extract_reward(
                    info0=info0,
                    vec_reward=vec_reward,
                    require_info_reward=not allow_reward_fallback,
                )
                rewards.append(reward)

                if get_done_bool(done):
                    break

            cumulative_rewards.append(discounted_sum(rewards, gamma))
            lengths.append(len(rewards))

    finally:
        close_env_safely(eval_env)

    return (
        np.asarray(cumulative_rewards, dtype=float),
        np.asarray(lengths, dtype=int),
        gamma,
    )


 
# Result lookup and saving
 
def find_row(rows, zeta_val, wn_val, lam_val):
    """Find one result row by zeta, omega_n, and lambda."""
    for row in rows:
        if (
            abs(row["zeta"] - zeta_val) < 1e-12
            and abs(row["wn"] - wn_val) < 1e-12
            and abs(row["lambda"] - lam_val) < 1e-12
        ):
            return row
    return None


def save_csv(rows, out_dir):
    """Save the main summary CSV."""
    csv_path = out_dir / "expected_cumulative_reward_by_lambda_summary.csv"

    fieldnames = [
        "zeta",
        "wn",
        "lambda",
        "run_id",
        "gamma",
        "n_rollouts",
        "full_expected_cumulative_reward_mean",
        "full_expected_cumulative_reward_std",
        "reduced_lambda0_expected_cumulative_reward_mean",
        "reduced_lambda0_expected_cumulative_reward_std",
        "full_minus_reduced_lambda0_mean",
        "full_minus_reduced_lambda0_std",
        "full_episode_length_mean",
        "full_episode_length_std",
        "reduced_lambda0_episode_length_mean",
        "reduced_lambda0_episode_length_std",
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


 
# Plotting
 
def save_expected_cumulative_reward_plot(
    rows,
    zeta_val,
    wn_values,
    lambda_values,
    out_dir,
    reduced_baseline,
    std_band=False,
):
    """Save the main curve plot."""
    wn_arr = np.asarray(wn_values, dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, lam_val in enumerate(lambda_values):
        means = []
        stds = []
        for wn_val in wn_values:
            row = find_row(rows, zeta_val=zeta_val, wn_val=wn_val, lam_val=lam_val)
            means.append(np.nan if row is None else row["full_expected_cumulative_reward_mean"])
            stds.append(np.nan if row is None else row["full_expected_cumulative_reward_std"])

        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)

        ax.plot(
            wn_arr,
            means,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.0,
            label=rf"Full deployment, {lambda_label(lam_val)}",
        )
        if std_band:
            ax.fill_between(wn_arr, means - stds, means + stds, alpha=0.15)

    baseline_mean = float(reduced_baseline["mean"])
    baseline_std = float(reduced_baseline["std"])
    ax.axhline(
        baseline_mean,
        linestyle="--",
        linewidth=2.0,
        label=rf"Reduced model, $\lambda=0$ baseline ({baseline_mean:+.1f})",
    )
    if std_band:
        ax.fill_between(
            [float(np.nanmin(wn_arr)), float(np.nanmax(wn_arr))],
            [baseline_mean - baseline_std, baseline_mean - baseline_std],
            [baseline_mean + baseline_std, baseline_mean + baseline_std],
            alpha=0.10,
        )

    ax.set_xlabel(r"Natural frequency $\omega_n$")
    ax.set_ylabel("Expected cumulative reward")
    ax.set_title(rf"Expected cumulative reward, $\zeta={zeta_val:g}$")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Policy")
    fig.tight_layout()

    path = out_dir / f"expected_cumulative_reward_by_lambda_zeta_{safe_float_name(zeta_val)}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


 
# Main
 
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot expected cumulative reward for multiple lambda policies under "
            "full-model deployment, with the reduced-model lambda=0 value shown "
            "as a baseline."
        )
    )

    parser.add_argument(
        "--lambda_values",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 3.0, 5.0],
        help="Lambda policies to evaluate. Default: 0 0.5 3 5.",
    )
    parser.add_argument(
        "--zeta_values",
        type=float,
        nargs="+",
        default=[0.7],
        help="Damping-ratio values. Example: --zeta_values 0.7",
    )
    parser.add_argument(
        "--wn_values",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        help="Natural-frequency values, matching the previous heatmap style.",
    )
    parser.add_argument(
        "--max_rollouts",
        type=int,
        default=None,
        help="Optional quick-test limit on the number of fixed initial states.",
    )
    parser.add_argument(
        "--episode_log_every",
        type=int,
        default=25,
        help=(
            "Print progress every N episodes. Use 1 to print every episode, "
            "or 0 to suppress episode-level progress. Default: 25."
        ),
    )
    parser.add_argument(
        "--std_band",
        action="store_true",
        help="Show mean +/- std bands around each curve. Default: off.",
    )
    parser.add_argument(
        "--allow_reward_fallback",
        action="store_true",
        help=(
            "Allow fallback to the environment reward if info['raw_r'] is missing. "
            "Use only if the environment reward is known to exclude theta-variation cost."
        ),
    )

    args = parser.parse_args()

    lambda_values = unique_floats(args.lambda_values)
    zeta_values = unique_floats(args.zeta_values)
    wn_values = unique_floats(args.wn_values)

    init_states = np.asarray(comp.INIT_STATES, dtype=np.float32)
    if args.max_rollouts is not None:
        if int(args.max_rollouts) <= 0:
            raise ValueError("--max_rollouts must be positive when provided.")
        init_states = init_states[: int(args.max_rollouts)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"expected_cumulative_reward_by_lambda_with_reduced_lambda0_baseline_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===================================================")
    print("Expected cumulative reward by lambda")
    print(f"lambda values = {lambda_values}")
    print(f"zeta values   = {zeta_values}")
    print(f"omega values  = {wn_values}")
    print(f"rollouts      = {len(init_states)}")
    print(f"output        = {out_dir}")
    print("baseline      = reduced-model lambda=0 expected cumulative reward")
    print("cost term     = not included")
    print("===================================================")

    # Reduced lambda=0 baseline 
    reduced_values, reduced_lengths, gamma_reduced = evaluate_reduced_lambda0_baseline(
        init_states=init_states,
        episode_log_every=args.episode_log_every,
        allow_reward_fallback=True,
    )
    reduced_mean, reduced_std = mean_std(reduced_values)
    reduced_len_mean, reduced_len_std = mean_std(reduced_lengths)
    reduced_baseline = {
        "lambda": 0.0,
        "run_id": lambda_to_run_id(0.0),
        "gamma": float(gamma_reduced),
        "mean": reduced_mean,
        "std": reduced_std,
        "length_mean": reduced_len_mean,
        "length_std": reduced_len_std,
    }

    print("\nReduced lambda=0 baseline summary:")
    print(f"  gamma             = {gamma_reduced:.6f}")
    print(f"  mean reward       = {reduced_mean:+.6f}")
    print(f"  std reward        = {reduced_std:.6f}")
    print(f"  length mean/std   = {reduced_len_mean:.2f} / {reduced_len_std:.2f}")

    rows = []

    for zeta_val in zeta_values:
        for lam_val in lambda_values:
            run_id = lambda_to_run_id(lam_val)
            for wn_val in wn_values:
                full_values, lengths, gamma = evaluate_full_expected_cumulative_reward(
                    run_id=run_id,
                    lam_val=lam_val,
                    wn_val=wn_val,
                    zeta_val=zeta_val,
                    init_states=init_states,
                    require_info_reward=not args.allow_reward_fallback,
                    episode_log_every=args.episode_log_every,
                )

                full_mean, full_std = mean_std(full_values)
                len_mean, len_std = mean_std(lengths)

                if len(full_values) == len(reduced_values):
                    diff_values = full_values - reduced_values
                    diff_mean, diff_std = mean_std(diff_values)
                else:
                    diff_mean = full_mean - reduced_mean
                    diff_std = float(np.sqrt(full_std ** 2 + reduced_std ** 2))

                row = {
                    "zeta": float(zeta_val),
                    "wn": float(wn_val),
                    "lambda": float(lam_val),
                    "run_id": run_id,
                    "gamma": float(gamma),
                    "n_rollouts": int(len(init_states)),
                    "full_expected_cumulative_reward_mean": full_mean,
                    "full_expected_cumulative_reward_std": full_std,
                    "reduced_lambda0_expected_cumulative_reward_mean": reduced_mean,
                    "reduced_lambda0_expected_cumulative_reward_std": reduced_std,
                    "full_minus_reduced_lambda0_mean": diff_mean,
                    "full_minus_reduced_lambda0_std": diff_std,
                    "full_episode_length_mean": len_mean,
                    "full_episode_length_std": len_std,
                    "reduced_lambda0_episode_length_mean": reduced_len_mean,
                    "reduced_lambda0_episode_length_std": reduced_len_std,
                }
                rows.append(row)

                print(
                    f"\nRESULT | zeta={zeta_val:g}, omega_n={wn_val:g}, lambda={lam_val:g} | "
                    f"mean reward={full_mean:+.6f}, std={full_std:.6f}, "
                    f"minus reduced lambda=0={diff_mean:+.6f}, len={len_mean:.2f}"
                )

    csv_path = save_csv(rows, out_dir)

    plot_paths = []
    for zeta_val in zeta_values:
        plot_paths.append(
            save_expected_cumulative_reward_plot(
                rows=rows,
                zeta_val=zeta_val,
                wn_values=wn_values,
                lambda_values=lambda_values,
                out_dir=out_dir,
                reduced_baseline=reduced_baseline,
                std_band=args.std_band,
            )
        )

    print("\n===================================================")
    print("Saved outputs")
    print(f"CSV: {csv_path}")
    for path in plot_paths:
        print(f"Plot: {path}")
    print("===================================================\n")


if __name__ == "__main__":
    main()
