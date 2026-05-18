# -*- coding: utf-8 -*-
"""
Evaluate deployed discounted reference variation J_K and plot lambda heatmaps.

This script is the J_K-variation analogue of evaluate_lambda_heatmap.py.
It evaluates lambda-trained policies in the deployed full/closed-loop model for
multiple inner-loop settings (omega_n, zeta), then saves:

    - final_jk_variation_summary_metrics.csv
    - final_jk_variation_diff_vs_lambda0.csv
    - one heatmap per zeta for
          J_K^var(lambda) - J_K^var(lambda=0)
    - optional absolute J_K^var heatmaps
    - optional line plots

Formula used in each deployed rollout:

    J_K^var(pi_lambda)
        = sum_{t=0}^{T-1} gamma^t ||theta_star_t - theta_star_{t-1}||_P,

with theta_star_{-1} = 0. For the current scalar theta-reference problem,

    ||theta_star_t - theta_star_{t-1}||_P
        = sqrt(P) * abs(theta_star_t - theta_star_{t-1}).

The reported value is the empirical mean over fixed initial states:

    Jhat_K^var(lambda; omega_n, zeta)
        = (1/N) sum_i J_{K,i}^var(lambda; omega_n, zeta).

The main heatmap plots:

    Delta Jhat_K^var(lambda)
        = Jhat_K^var(lambda) - Jhat_K^var(lambda=0).

Color convention for the difference heatmap:
    green  : smaller deployed reference variation than lambda=0  (good)
    yellow : close to lambda=0
    red    : larger deployed reference variation than lambda=0   (bad)

Put this file in:
    scripts/evaluate_lambda_jk_heatmap.py

Run from the project root, for example:

    python scripts/evaluate_lambda_jk_heatmap.py --zeta_values 0.4 0.7 1.0 --wn_values 2 4 6 8 10 12 --lambdas 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10

Quick test:

    python scripts/evaluate_lambda_jk_heatmap.py --zeta_values 0.7 --wn_values 4 8 --lambdas 0 4.5 5 5.5 --max_rollouts 10

With absolute J_K heatmaps too:

    python scripts/evaluate_lambda_jk_heatmap.py --zeta_values 0.7 --wn_values 2 4 6 8 10 12 --lambdas 0 1 2 3 4 5 6 7 8 9 10 --absolute_heatmap

With line plots:

    python scripts/evaluate_lambda_jk_heatmap.py --zeta_values 0.7 --wn_values 2 4 6 8 10 12 --lambdas 0 1 2 3 4 5 6 7 8 9 10 --line_plot --plot_std
"""

import argparse
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Reuse your trusted project/evaluation helpers.
# This file should sit in scripts/, next to evaluate_lambda_comparison.py.
import evaluate_lambda_comparison as comp


# ------------------------------------------------------------
# Naming helpers
# ------------------------------------------------------------
def lambda_to_run_id(lam_val):
    """
    Convert lambda value to the saved model/run folder name.

    Examples:
        0.0  -> lambda_0
        0.5  -> lambda_0.5
        4.0  -> lambda_4
        4.8  -> lambda_4.8
        5.2  -> lambda_5.2

    This assumes your saved models/vectors are named in this format.
    """
    return f"lambda_{float(lam_val):g}"


def lambda_to_label(lam_val):
    return rf"$\lambda={float(lam_val):g}$"


def unique_floats(values):
    """Remove duplicate float values while preserving order."""
    result = []

    for val in values:
        val = float(val)
        if not any(abs(val - existing) < 1e-12 for existing in result):
            result.append(val)

    return result


def safe_float_name(val):
    """Make a float safe for filenames."""
    return f"{float(val):g}".replace("-", "m").replace(".", "p")


# ------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------
def combine_std_for_difference(std_a, std_b):
    """
    Approximate std of A - B using sqrt(std(A)^2 + std(B)^2).

    This is conservative. A paired std would be better if we stored
    per-rollout J_K(lambda) - J_K(0) using the same initial states.
    """
    if np.isnan(std_a) or np.isnan(std_b):
        return np.nan

    return float(np.sqrt(std_a ** 2 + std_b ** 2))


def compute_discounted_variation(theta_star_seq, gamma, p_weight=1.0):
    """
    Compute the scalar-theta version of

        sum_t gamma^t ||theta_star_t - theta_star_{t-1}||_P,

    with theta_star_{-1} = 0.

    For scalar theta and scalar P > 0,
        ||delta||_P = sqrt(P) * |delta|.
    """
    theta_star_seq = np.asarray(theta_star_seq, dtype=float).reshape(-1)

    if theta_star_seq.size == 0:
        return 0.0

    if p_weight <= 0.0:
        raise ValueError(f"p_weight must be positive, got {p_weight}.")

    prev = np.concatenate([[0.0], theta_star_seq[:-1]])
    increments = np.sqrt(float(p_weight)) * np.abs(theta_star_seq - prev)
    powers = float(gamma) ** np.arange(theta_star_seq.size, dtype=float)

    return float(np.dot(powers, increments))


# ------------------------------------------------------------
# Rollout evaluation for deployed J_K variation
# ------------------------------------------------------------
def run_rollouts_jk_variation(
    run_id,
    lam_val,
    wn_val,
    zeta_val,
    init_states,
    p_weight=1.0,
    use_info_cost=False,
):
    """
    Evaluate one trained lambda policy under the deployed closed-loop model.

    The deployed model is built by comp._build_eval_env_early(...), which uses
    the full second-order inner-loop environment with the chosen (omega_n, zeta).

    Returns a dictionary with mean/std of deployed discounted reference variation:

        J_K^var = sum_t gamma^t ||theta_star_t - theta_star_{t-1}||_P.

    If use_info_cost=True, the script uses info['cost'] from your wrapper.
    Otherwise, it recomputes the variation directly from the policy action.
    Direct recomputation is the default because it makes the formula explicit.
    """
    init_states = np.asarray(init_states, dtype=np.float32)
    n_rollouts = int(init_states.shape[0])

    eval_env = comp._build_eval_env_early(
        run_id=run_id,
        wn_val=float(wn_val),
        zeta_val=float(zeta_val),
        lam_val=float(lam_val),
        init_states=init_states,
    )

    model_path = comp._find_model_zip(run_id)
    model = comp.PPO.load(model_path, env=eval_env, device="cpu")

    inner = eval_env.venv.envs[0]
    base = comp._unwrap_to_base_env(inner)
    max_steps_ep = int(
        getattr(base, "max_steps", getattr(comp.HEARLY, "MAX_STEPS_EP", 400))
    )
    gamma_local = float(getattr(model, "gamma", getattr(comp.HEARLY, "GAMMA", 0.994)))

    jk_values = []
    episode_lengths = []

    print(
        f"\n[RUN CONFIG] deployed J_K variation | "
        f"run_id={run_id}, lambda={lam_val:g}, "
        f"omega_n={wn_val:g}, zeta={zeta_val:g}, "
        f"rollouts={n_rollouts}"
    )

    for ep in range(n_rollouts):
        print(f"  Episode {ep + 1}/{n_rollouts}")
        obs = eval_env.reset()

        theta_star_seq = []
        info_cost_seq = []

        for k in range(max_steps_ep):
            action, _ = model.predict(obs, deterministic=True)
            a_flat = np.asarray(action, dtype=float).reshape(-1)
            theta_star_now = float(a_flat[1]) if a_flat.size >= 2 else 0.0
            theta_star_seq.append(theta_star_now)

            obs, reward, done, info = eval_env.step(action)
            info0 = info[0] if isinstance(info, (list, tuple)) else info

            if use_info_cost:
                info_cost_seq.append(float(info0.get("cost", 0.0)))

            done_bool = bool(np.asarray(done).reshape(-1)[0])
            if done_bool:
                break

        theta_star_seq = np.asarray(theta_star_seq, dtype=float)
        episode_lengths.append(int(theta_star_seq.size))

        if use_info_cost:
            cost_seq = np.asarray(info_cost_seq, dtype=float).reshape(-1)
            powers = gamma_local ** np.arange(cost_seq.size, dtype=float)
            jk_ep = float(np.dot(powers, np.sqrt(float(p_weight)) * cost_seq))
        else:
            jk_ep = compute_discounted_variation(
                theta_star_seq=theta_star_seq,
                gamma=gamma_local,
                p_weight=p_weight,
            )

        jk_values.append(jk_ep)

    jk_values = np.asarray(jk_values, dtype=float)
    episode_lengths = np.asarray(episode_lengths, dtype=float)

    return {
        "jk_variation_values": jk_values,
        "jk_variation_mean": float(np.nanmean(jk_values)),
        "jk_variation_std": float(np.nanstd(jk_values)),
        "episode_length_mean": float(np.nanmean(episode_lengths)),
        "episode_length_std": float(np.nanstd(episode_lengths)),
        "gamma": gamma_local,
        "p_weight": float(p_weight),
        "n_rollouts": n_rollouts,
    }


# ------------------------------------------------------------
# Evaluation over zeta, omega_n, lambda
# ------------------------------------------------------------
def evaluate_all_jk_variation(
    zeta_values,
    wn_values,
    runs,
    init_states,
    p_weight=1.0,
    use_info_cost=False,
):
    rows = []

    for zeta_eval in zeta_values:
        for wn_val in wn_values:
            print(
                f"\n===== Evaluating deployed J_K variation: "
                f"omega_n={wn_val:g}, zeta={zeta_eval:g} ====="
            )

            for run_id, lam_val, lam_label in runs:
                metrics = run_rollouts_jk_variation(
                    run_id=run_id,
                    lam_val=lam_val,
                    wn_val=wn_val,
                    zeta_val=zeta_eval,
                    init_states=init_states,
                    p_weight=p_weight,
                    use_info_cost=use_info_cost,
                )

                row = {
                    "zeta": float(zeta_eval),
                    "wn": float(wn_val),
                    "lambda": float(lam_val),
                    "run_id": run_id,
                    "final_deployed_jk_variation_mean": metrics["jk_variation_mean"],
                    "final_deployed_jk_variation_std": metrics["jk_variation_std"],
                    "episode_length_mean": metrics["episode_length_mean"],
                    "episode_length_std": metrics["episode_length_std"],
                    "gamma": metrics["gamma"],
                    "p_weight": metrics["p_weight"],
                    "n_rollouts": metrics["n_rollouts"],
                }

                rows.append(row)

                print(
                    f"lambda={lam_val:>5g} | "
                    f"run_id={run_id:<12s} | "
                    f"J_K_var={row['final_deployed_jk_variation_mean']:.6f} "
                    f"+/- {row['final_deployed_jk_variation_std']:.6f} | "
                    f"episode_len={row['episode_length_mean']:.1f}"
                )

    return rows


# ------------------------------------------------------------
# Difference relative to lambda=0
# ------------------------------------------------------------
def build_jk_diff_rows(rows, zeta_values, wn_values):
    diff_rows = []

    for zeta_eval in zeta_values:
        for wn_val in wn_values:
            base_mean = None
            base_std = None

            for row in rows:
                if (
                    abs(float(row["zeta"]) - float(zeta_eval)) < 1e-12
                    and abs(float(row["wn"]) - float(wn_val)) < 1e-12
                    and abs(float(row["lambda"])) < 1e-12
                ):
                    base_mean = float(row["final_deployed_jk_variation_mean"])
                    base_std = float(row["final_deployed_jk_variation_std"])
                    break

            if base_mean is None:
                print(
                    f"Warning: no lambda=0 baseline found for "
                    f"zeta={zeta_eval:g}, omega_n={wn_val:g}. Skipping."
                )
                continue

            diff_row = {
                "zeta": float(zeta_eval),
                "wn": float(wn_val),
                "jk_variation_lambda_0_mean": base_mean,
                "jk_variation_lambda_0_std": base_std,
            }

            for row in rows:
                if (
                    abs(float(row["zeta"]) - float(zeta_eval)) < 1e-12
                    and abs(float(row["wn"]) - float(wn_val)) < 1e-12
                ):
                    lam = float(row["lambda"])
                    lam_key = f"{lam:g}"

                    abs_mean_key = f"jk_variation_lambda_{lam_key}_mean"
                    abs_std_key = f"jk_variation_lambda_{lam_key}_std"
                    diff_mean_key = f"jk_variation_lambda_{lam_key}_minus_lambda_0_mean"
                    diff_std_key = f"jk_variation_lambda_{lam_key}_minus_lambda_0_std_approx"

                    jk_mean = float(row["final_deployed_jk_variation_mean"])
                    jk_std = float(row["final_deployed_jk_variation_std"])

                    diff_row[abs_mean_key] = jk_mean
                    diff_row[abs_std_key] = jk_std
                    diff_row[diff_mean_key] = jk_mean - base_mean

                    if abs(lam) < 1e-12:
                        diff_row[diff_std_key] = 0.0
                    else:
                        diff_row[diff_std_key] = combine_std_for_difference(
                            jk_std,
                            base_std,
                        )

            diff_rows.append(diff_row)

    return diff_rows


# ------------------------------------------------------------
# CSV saving
# ------------------------------------------------------------
def save_rows_csv(rows, path):
    if not rows:
        raise RuntimeError(f"No rows to save for {path}")

    headers = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_diff_csv(diff_rows, runs, path):
    if not diff_rows:
        raise RuntimeError(f"No diff rows to save for {path}")

    headers = ["zeta", "wn"]

    for _, lam_val, _ in runs:
        lam_key = f"{float(lam_val):g}"
        keys = [
            f"jk_variation_lambda_{lam_key}_mean",
            f"jk_variation_lambda_{lam_key}_std",
            f"jk_variation_lambda_{lam_key}_minus_lambda_0_mean",
            f"jk_variation_lambda_{lam_key}_minus_lambda_0_std_approx",
        ]

        for key in keys:
            if key not in headers:
                headers.append(key)

    # Keep any extra columns, such as baseline convenience columns.
    for row in diff_rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(diff_rows)


# ------------------------------------------------------------
# Heatmap plotting: difference relative to lambda=0
# ------------------------------------------------------------
def plot_jk_diff_heatmaps_by_zeta(
    diff_rows,
    lambda_values,
    wn_values,
    zeta_values,
    out_dir,
    annotate=True,
):
    """
    Creates one heatmap per zeta for

        J_K^var(lambda) - J_K^var(lambda=0).

    Negative values mean the lambda-policy has smaller deployed reference
    variation than lambda=0. Therefore, the colormap is reversed so that
    negative values are green and positive values are red.
    """
    lambda_values = [float(v) for v in lambda_values]
    wn_values = [float(v) for v in wn_values]
    zeta_values = [float(v) for v in zeta_values]

    lookup = {}
    all_vals = []

    for row in diff_rows:
        zeta = float(row["zeta"])
        wn = float(row["wn"])

        for lam in lambda_values:
            key = f"jk_variation_lambda_{lam:g}_minus_lambda_0_mean"

            if key in row:
                val = float(row[key])
                lookup[(zeta, lam, wn)] = val

                if not np.isnan(val):
                    all_vals.append(val)

    if len(all_vals) == 0:
        print("No J_K variation-difference values found for heatmap.")
        return

    max_abs = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals)))

    if max_abs < 1e-12:
        max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    for zeta in zeta_values:
        mat = np.full((len(lambda_values), len(wn_values)), np.nan)

        for i, lam in enumerate(lambda_values):
            for j, wn in enumerate(wn_values):
                mat[i, j] = lookup.get((zeta, lam, wn), np.nan)

        plt.figure(figsize=(1.0 * len(wn_values) + 4, 0.45 * len(lambda_values) + 3))

        im = plt.imshow(
            mat,
            aspect="auto",
            cmap="RdYlGn_r",
            norm=norm,
            origin="lower",
        )

        plt.xticks(
            ticks=np.arange(len(wn_values)),
            labels=[f"{v:g}" for v in wn_values],
        )

        plt.yticks(
            ticks=np.arange(len(lambda_values)),
            labels=[f"{v:g}" for v in lambda_values],
        )

        plt.xlabel(r"$\omega_n$")
        plt.ylabel(r"$\lambda$")
        plt.title(
            "Deployed reference-variation difference relative to lambda=0\n"
            rf"$\widehat{{J}}^{{\mathrm{{var}}}}_K(\lambda)"
            rf"-\widehat{{J}}^{{\mathrm{{var}}}}_K(0)$, "
            rf"$\zeta={zeta:g}$"
        )

        cbar = plt.colorbar(im)
        cbar.set_label(
            r"$\widehat{J}^{\mathrm{var}}_K(\lambda)"
            r"-\widehat{J}^{\mathrm{var}}_K(0)$"
            "  (negative = smoother than lambda=0)"
        )

        if annotate:
            for i in range(len(lambda_values)):
                for j in range(len(wn_values)):
                    val = mat[i, j]
                    if not np.isnan(val):
                        plt.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

        plt.tight_layout()

        heatmap_path = out_dir / f"heatmap_jk_variation_diff_vs_lambda0_zeta_{safe_float_name(zeta)}.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"  {heatmap_path}")


# ------------------------------------------------------------
# Heatmap plotting: absolute deployed J_K variation
# ------------------------------------------------------------
def plot_jk_absolute_heatmaps_by_zeta(
    diff_rows,
    lambda_values,
    wn_values,
    zeta_values,
    out_dir,
    annotate=True,
):
    """
    Creates one heatmap per zeta for absolute deployed J_K variation:

        J_K^var(lambda).

    Smaller values mean smoother deployed reference commands.
    """
    lambda_values = [float(v) for v in lambda_values]
    wn_values = [float(v) for v in wn_values]
    zeta_values = [float(v) for v in zeta_values]

    lookup = {}
    all_vals = []

    for row in diff_rows:
        zeta = float(row["zeta"])
        wn = float(row["wn"])

        for lam in lambda_values:
            key = f"jk_variation_lambda_{lam:g}_mean"

            if key in row:
                val = float(row[key])
                lookup[(zeta, lam, wn)] = val

                if not np.isnan(val):
                    all_vals.append(val)

    if len(all_vals) == 0:
        print("No absolute J_K variation values found for heatmap.")
        return

    for zeta in zeta_values:
        mat = np.full((len(lambda_values), len(wn_values)), np.nan)

        for i, lam in enumerate(lambda_values):
            for j, wn in enumerate(wn_values):
                mat[i, j] = lookup.get((zeta, lam, wn), np.nan)

        plt.figure(figsize=(1.0 * len(wn_values) + 4, 0.45 * len(lambda_values) + 3))

        im = plt.imshow(
            mat,
            aspect="auto",
            cmap="viridis_r",
            origin="lower",
        )

        plt.xticks(
            ticks=np.arange(len(wn_values)),
            labels=[f"{v:g}" for v in wn_values],
        )

        plt.yticks(
            ticks=np.arange(len(lambda_values)),
            labels=[f"{v:g}" for v in lambda_values],
        )

        plt.xlabel(r"$\omega_n$")
        plt.ylabel(r"$\lambda$")
        plt.title(
            "Absolute deployed discounted reference variation\n"
            rf"$\widehat{{J}}^{{\mathrm{{var}}}}_K(\lambda)$, "
            rf"$\zeta={zeta:g}$"
        )

        cbar = plt.colorbar(im)
        cbar.set_label(r"$\widehat{J}^{\mathrm{var}}_K(\lambda)$")

        if annotate:
            for i in range(len(lambda_values)):
                for j in range(len(wn_values)):
                    val = mat[i, j]
                    if not np.isnan(val):
                        plt.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

        plt.tight_layout()

        heatmap_path = out_dir / f"heatmap_jk_variation_absolute_zeta_{safe_float_name(zeta)}.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"  {heatmap_path}")


# ------------------------------------------------------------
# Optional line plots
# ------------------------------------------------------------
def plot_jk_line_plots_by_zeta(
    diff_rows,
    runs,
    wn_values,
    zeta_values,
    out_dir,
    plot_std=False,
):
    for zeta in zeta_values:
        rows_for_zeta = [
            row for row in diff_rows
            if abs(float(row["zeta"]) - float(zeta)) < 1e-12
        ]

        if not rows_for_zeta:
            continue

        plt.figure(figsize=(10, 5.5))

        xs0 = [row["wn"] for row in rows_for_zeta]
        ys0 = [0.0 for _ in rows_for_zeta]

        plt.plot(
            xs0,
            ys0,
            linestyle=":",
            marker="o",
            label=r"$\lambda=0$ baseline",
        )

        for _, lam_val, lam_label in runs:
            if abs(float(lam_val)) < 1e-12:
                continue

            mean_key = f"jk_variation_lambda_{float(lam_val):g}_minus_lambda_0_mean"
            std_key = f"jk_variation_lambda_{float(lam_val):g}_minus_lambda_0_std_approx"

            xs = []
            ys = []
            yerrs = []

            for row in rows_for_zeta:
                if mean_key in row:
                    xs.append(row["wn"])
                    ys.append(row[mean_key])
                    yerrs.append(row.get(std_key, np.nan))

            if len(xs) == 0:
                continue

            if plot_std and not np.all(np.isnan(yerrs)):
                plt.errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    marker="o",
                    capsize=3,
                    label=lam_label + r" $-$ $\lambda=0$",
                )
            else:
                plt.plot(
                    xs,
                    ys,
                    marker="o",
                    label=lam_label + r" $-$ $\lambda=0$",
                )

        plt.axhline(0.0, linewidth=1, alpha=0.8, linestyle="--")

        plt.xlabel(r"$\omega_n$")
        plt.ylabel(
            "Delta deployed discounted reference variation\n"
            r"($\widehat{J}^{\mathrm{var}}_K(\lambda)"
            r"-\widehat{J}^{\mathrm{var}}_K(0)$)"
        )

        plt.title(
            "Deployed reference-variation difference relative to lambda=0\n"
            rf"for different $\omega_n$, $\zeta={zeta:g}$"
        )

        plt.legend()
        plt.tight_layout()

        plot_path = out_dir / f"line_jk_variation_diff_vs_lambda0_zeta_{safe_float_name(zeta)}.png"
        plt.savefig(plot_path, dpi=300)
        plt.show()

        print(f"  {plot_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run deployed J_K reference-variation comparison and generate heatmaps."
    )

    parser.add_argument(
        "--zeta",
        type=float,
        default=0.7,
        help="Single damping ratio zeta. Used only if --zeta_values is not provided.",
    )

    parser.add_argument(
        "--zeta_values",
        type=float,
        nargs="+",
        default=None,
        help="List of zeta values to evaluate.",
    )

    parser.add_argument(
        "--wn_values",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        help="List of omega_n values to evaluate.",
    )

    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
        ],
        help=(
            "List of lambda values to evaluate. "
            "The script automatically looks for run IDs named lambda_<value>."
        ),
    )

    parser.add_argument(
        "--max_rollouts",
        type=int,
        default=None,
        help="Optional limit on number of initial states. Useful for quick tests.",
    )

    parser.add_argument(
        "--p_weight",
        type=float,
        default=1.0,
        help=(
            "Scalar P weight for ||delta theta_star||_P. "
            "For scalar theta, norm = sqrt(P) * abs(delta)."
        ),
    )

    parser.add_argument(
        "--use_info_cost",
        action="store_true",
        help=(
            "Use info['cost'] from RewardCostWrapper instead of recomputing "
            "|theta_star_t - theta_star_{t-1}| from actions."
        ),
    )

    parser.add_argument(
        "--no_heatmap",
        action="store_true",
        help="Do not create heatmaps; save CSV files only.",
    )

    parser.add_argument(
        "--absolute_heatmap",
        action="store_true",
        help="Also create heatmaps for absolute deployed J_K variation.",
    )

    parser.add_argument(
        "--no_annotate_heatmap",
        action="store_true",
        help="Do not write numeric values inside heatmap cells.",
    )

    parser.add_argument(
        "--line_plot",
        action="store_true",
        help="Also create line plots of J_K variation difference versus omega_n.",
    )

    parser.add_argument(
        "--plot_std",
        action="store_true",
        help="Add approximate std error bars to the optional line plots.",
    )

    args = parser.parse_args()

    if args.zeta_values is None:
        zeta_values = [float(args.zeta)]
    else:
        zeta_values = unique_floats(args.zeta_values)

    wn_values = unique_floats(args.wn_values)
    lambda_values = unique_floats(args.lambdas)

    # lambda=0 is required as the baseline.
    if not any(abs(lam) < 1e-12 for lam in lambda_values):
        print("lambda=0 was not provided, so it is added automatically as baseline.")
        lambda_values = [0.0] + lambda_values

    runs = [
        (lambda_to_run_id(lam_val), lam_val, lambda_to_label(lam_val))
        for lam_val in lambda_values
    ]

    init_states = comp.INIT_STATES

    if args.max_rollouts is not None:
        init_states = init_states[: args.max_rollouts]
        comp.N_ROLLOUTS = len(init_states)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = comp.PROJECT_ROOT / "results" / f"lambda_jk_variation_heatmap_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===================================================")
    print("Deployed J_K reference-variation heatmap comparison")
    print(f"zeta values   = {zeta_values}")
    print(f"wn values     = {wn_values}")
    print(f"lambdas       = {lambda_values}")
    print(f"run ids       = {[run_id for run_id, _, _ in runs]}")
    print(f"rollouts      = {len(init_states)}")
    print(f"p_weight      = {args.p_weight}")
    print(f"use_info_cost = {args.use_info_cost}")
    print(f"output        = {out_dir}")
    print("Formula       = sum_t gamma^t ||theta_star_t - theta_star_{t-1}||_P")
    print("Convention    = theta_star_{-1} = 0")
    print("===================================================\n")

    rows = evaluate_all_jk_variation(
        zeta_values=zeta_values,
        wn_values=wn_values,
        runs=runs,
        init_states=init_states,
        p_weight=args.p_weight,
        use_info_cost=args.use_info_cost,
    )

    if not rows:
        raise RuntimeError("No evaluation rows were generated.")

    diff_rows = build_jk_diff_rows(
        rows=rows,
        zeta_values=zeta_values,
        wn_values=wn_values,
    )

    if not diff_rows:
        raise RuntimeError("No J_K variation-difference rows were generated.")

    summary_csv = out_dir / "final_jk_variation_summary_metrics.csv"
    diff_csv = out_dir / "final_jk_variation_diff_vs_lambda0.csv"

    save_rows_csv(rows, summary_csv)
    save_diff_csv(diff_rows, runs, diff_csv)

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {diff_csv}")

    if not args.no_heatmap:
        plot_jk_diff_heatmaps_by_zeta(
            diff_rows=diff_rows,
            lambda_values=lambda_values,
            wn_values=wn_values,
            zeta_values=zeta_values,
            out_dir=out_dir,
            annotate=not args.no_annotate_heatmap,
        )

        if args.absolute_heatmap:
            plot_jk_absolute_heatmaps_by_zeta(
                diff_rows=diff_rows,
                lambda_values=lambda_values,
                wn_values=wn_values,
                zeta_values=zeta_values,
                out_dir=out_dir,
                annotate=not args.no_annotate_heatmap,
            )

    if args.line_plot:
        plot_jk_line_plots_by_zeta(
            diff_rows=diff_rows,
            runs=runs,
            wn_values=wn_values,
            zeta_values=zeta_values,
            out_dir=out_dir,
            plot_std=args.plot_std,
        )


if __name__ == "__main__":
    main()
