# -*- coding: utf-8 -*-
"""
Compact lambda-comparison evaluator with heatmaps.


It evaluates selected lambda policies over selected omega_n and zeta values,
then saves:
    - final_summary_metrics.csv
    - final_reward_diff_vs_lambda0.csv
    - one heatmap per zeta
    - optional line plots
    - optional std/error bars in line plots

Main heatmap:
    x-axis: omega_n
    y-axis: lambda
    color : J_rew(lambda) - J_rew(lambda=0)

Color meaning:
    red    : worse than lambda=0
    yellow : close to lambda=0
    green  : better than lambda=0

Example:
    python scripts/evaluate_lambda_summary.py --zeta_values 0.4 0.7 1.0 --wn_values 2 4 6 8 10 12 --lambdas 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10

Quick test:
    python scripts/evaluate_lambda_summary.py --zeta_values 0.7 --wn_values 4 8 --lambdas 0 4.5 5 5.5 --max_rollouts 10

With line plots:
    python scripts/evaluate_lambda_summary.py --zeta_values 0.4 0.7 1.0 --wn_values 2 4 6 8 10 12 --lambdas 0 1 2 3 4 5 6 7 8 9 10 --line_plot

With line plots and std:
    python scripts/evaluate_lambda_summary.py --zeta_values 0.7 --wn_values 2 4 6 8 10 12 --lambdas 0 1 2 3 4 5 6 7 8 9 10 --line_plot --plot_std
"""

import argparse
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Import comprehensive comparison code
import evaluate_lambda_comparison as comp


  
# Naming helpers
  
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

    If your folders use lambda_0_5 instead of lambda_0.5,
    replace the return line with:
        return f"lambda_{lam_val:g}".replace(".", "_")
    """
    return f"lambda_{lam_val:g}"


def lambda_to_label(lam_val):
    return rf"$\lambda={lam_val:g}$"


def unique_floats(values):
    """
    Remove duplicate float values while preserving order.
    """
    result = []

    for val in values:
        val = float(val)

        if not any(abs(val - existing) < 1e-12 for existing in result):
            result.append(val)

    return result


  
# Metric helpers
  
def last_valid_value(arr):
    arr = np.asarray(arr, dtype=float)
    valid = ~np.isnan(arr)

    if not np.any(valid):
        return np.nan

    return float(arr[valid][-1])


def last_valid_metric(metrics, key):
    if key not in metrics:
        return np.nan

    return last_valid_value(metrics[key])


def last_valid_std(metrics, mean_key):
    """
    Safely get the std array corresponding to a mean array.

    Example:
        cum_rew_mean -> cum_rew_std
    """
    std_key = mean_key.replace("_mean", "_std")

    if std_key not in metrics:
        return np.nan

    return last_valid_value(metrics[std_key])


def combine_std_for_difference(std_a, std_b):
    """
    Approximate std of A - B using:
        std(A - B) approx sqrt(std(A)^2 + std(B)^2)

    This is conservative if the same initial states are used for both policies.
    A paired std would be better, but that requires per-rollout rewards.
    """
    if np.isnan(std_a) or np.isnan(std_b):
        return np.nan

    return float(np.sqrt(std_a**2 + std_b**2))


  
# Evaluation
  
def evaluate_all(zeta_values, wn_values, runs, init_states):
    rows = []

    for zeta_eval in zeta_values:
        for wn_val in wn_values:
            print(
                f"\n===== Evaluating omega_n = {wn_val:g}, "
                f"zeta = {zeta_eval:g} ====="
            )

            for run_id, lam_val, lam_label in runs:
                metrics = comp.run_rollouts_metrics_early(
                    run_id=run_id,
                    lam_val=lam_val,
                    wn_val=wn_val,
                    zeta_val=zeta_eval,
                    init_states=init_states,
                )

                row = {
                    "zeta": zeta_eval,
                    "wn": wn_val,
                    "lambda": lam_val,
                    "run_id": run_id,

                    "final_cum_discounted_reward_mean": last_valid_metric(
                        metrics, "cum_rew_mean"
                    ),
                    "final_cum_discounted_reward_std": last_valid_std(
                        metrics, "cum_rew_mean"
                    ),

                    "final_cum_discounted_return_mean": last_valid_metric(
                        metrics, "cum_ret_mean"
                    ),
                    "final_cum_discounted_return_std": last_valid_std(
                        metrics, "cum_ret_mean"
                    ),

                    "final_position_error_mean": last_valid_metric(
                        metrics, "pos_err_mean"
                    ),
                    "final_position_error_std": last_valid_std(
                        metrics, "pos_err_mean"
                    ),

                    "final_tracking_pre_mean": last_valid_metric(
                        metrics, "cum_tracking_pre_mean"
                    ),
                    "final_tracking_pre_std": last_valid_std(
                        metrics, "cum_tracking_pre_mean"
                    ),

                    "final_tracking_post_mean": last_valid_metric(
                        metrics, "cum_tracking_post_mean"
                    ),
                    "final_tracking_post_std": last_valid_std(
                        metrics, "cum_tracking_post_mean"
                    ),

                    "final_x_mean": last_valid_metric(metrics, "x_mean"),
                    "final_x_std": last_valid_std(metrics, "x_mean"),

                    "final_z_mean": last_valid_metric(metrics, "z_mean"),
                    "final_z_std": last_valid_std(metrics, "z_mean"),

                    "final_vx_mean": last_valid_metric(metrics, "vx_mean"),
                    "final_vx_std": last_valid_std(metrics, "vx_mean"),

                    "final_vz_mean": last_valid_metric(metrics, "vz_mean"),
                    "final_vz_std": last_valid_std(metrics, "vz_mean"),
                }

                rows.append(row)

                print(
                    f"lambda={lam_val:>5g} | "
                    f"run_id={run_id:<12s} | "
                    f"reward={row['final_cum_discounted_reward_mean']:+.4f} "
                    f"+/- {row['final_cum_discounted_reward_std']:.4f} | "
                    f"return={row['final_cum_discounted_return_mean']:+.4f} | "
                    f"pos_err={row['final_position_error_mean']:.4f}"
                )

    return rows


  
# Difference relative to lambda=0
  
def build_diff_rows(rows, zeta_values, wn_values):
    diff_rows = []

    for zeta_eval in zeta_values:
        for wn_val in wn_values:
            base_mean = None
            base_std = None

            for row in rows:
                if (
                    abs(row["zeta"] - zeta_eval) < 1e-12
                    and abs(row["wn"] - wn_val) < 1e-12
                    and abs(row["lambda"]) < 1e-12
                ):
                    base_mean = row["final_cum_discounted_reward_mean"]
                    base_std = row["final_cum_discounted_reward_std"]
                    break

            if base_mean is None:
                print(
                    f"Warning: no lambda=0 baseline found for "
                    f"zeta={zeta_eval:g}, omega_n={wn_val:g}. Skipping."
                )
                continue

            diff_row = {
                "zeta": zeta_eval,
                "wn": wn_val,
            }

            for row in rows:
                if (
                    abs(row["zeta"] - zeta_eval) < 1e-12
                    and abs(row["wn"] - wn_val) < 1e-12
                ):
                    lam = row["lambda"]

                    mean_key = f"reward_lambda_{lam:g}_minus_lambda_0_mean"
                    std_key = f"reward_lambda_{lam:g}_minus_lambda_0_std_approx"

                    diff_row[mean_key] = (
                        row["final_cum_discounted_reward_mean"] - base_mean
                    )

                    if abs(lam) < 1e-12:
                        diff_row[std_key] = 0.0
                    else:
                        diff_row[std_key] = combine_std_for_difference(
                            row["final_cum_discounted_reward_std"],
                            base_std,
                        )

            diff_rows.append(diff_row)

    return diff_rows


  
# CSV saving
  
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
        mean_key = f"reward_lambda_{lam_val:g}_minus_lambda_0_mean"
        std_key = f"reward_lambda_{lam_val:g}_minus_lambda_0_std_approx"

        if mean_key not in headers:
            headers.append(mean_key)

        if std_key not in headers:
            headers.append(std_key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(diff_rows)


  
# Heatmap plotting
  
def plot_heatmaps_by_zeta(
    diff_rows,
    lambda_values,
    wn_values,
    zeta_values,
    out_dir,
    annotate=True,
):
    """
    Creates one heatmap per zeta.

    Rows:
        lambda values

    Columns:
        omega_n values

    Color:
        J_rew(lambda) - J_rew(lambda=0)
    """
    lambda_values = [float(v) for v in lambda_values]
    wn_values = [float(v) for v in wn_values]
    zeta_values = [float(v) for v in zeta_values]

    # Build lookup and global color range.
    lookup = {}
    all_vals = []

    for row in diff_rows:
        zeta = float(row["zeta"])
        wn = float(row["wn"])

        for lam in lambda_values:
            key = f"reward_lambda_{lam:g}_minus_lambda_0_mean"

            if key in row:
                val = float(row[key])
                lookup[(zeta, lam, wn)] = val

                if not np.isnan(val):
                    all_vals.append(val)

    if len(all_vals) == 0:
        print("No values found for heatmap.")
        return

    max_abs = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals)))

    if max_abs < 1e-12:
        max_abs = 1.0

    # Center yellow at zero.
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
            cmap="RdYlGn",
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
            "Reward improvement relative to lambda=0\n"
            rf"$J_{{\mathrm{{rew}}}}(\lambda)-J_{{\mathrm{{rew}}}}(0)$, "
            rf"$\zeta={zeta:g}$"
        )

        cbar = plt.colorbar(im)
        cbar.set_label(
            r"$J_{\mathrm{rew}}(\lambda)-J_{\mathrm{rew}}(\lambda=0)$"
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

        heatmap_path = out_dir / f"heatmap_reward_diff_vs_lambda0_zeta_{zeta:g}.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"  {heatmap_path}")


  
# Optional line plots
  
def plot_line_plots_by_zeta(
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
            if abs(lam_val) < 1e-12:
                continue

            mean_key = f"reward_lambda_{lam_val:g}_minus_lambda_0_mean"
            std_key = f"reward_lambda_{lam_val:g}_minus_lambda_0_std_approx"

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
            "Delta final cumulative discounted reward\n"
            r"($J_{\mathrm{rew}}(\lambda)-J_{\mathrm{rew}}(\lambda=0)$)"
        )

        plt.title(
            "Reward improvement relative to lambda=0\n"
            rf"for different $\omega_n$, $\zeta={zeta:g}$"
        )

        plt.legend()
        plt.tight_layout()

        plot_path = out_dir / f"line_reward_diff_vs_lambda0_zeta_{zeta:g}.png"
        plt.savefig(plot_path, dpi=300)
        plt.show()

        print(f"  {plot_path}")


  
# Main
  
def main():
    parser = argparse.ArgumentParser(
        description="Run compact lambda comparison and generate heatmaps."
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
        "--no_heatmap",
        action="store_true",
        help="Do not create heatmaps; save CSV files only.",
    )

    parser.add_argument(
        "--no_annotate_heatmap",
        action="store_true",
        help="Do not write numeric values inside heatmap cells.",
    )

    parser.add_argument(
        "--line_plot",
        action="store_true",
        help="Also create line plots of reward difference versus omega_n.",
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

    # Optional fast/debug mode.
    init_states = comp.INIT_STATES

    if args.max_rollouts is not None:
        init_states = init_states[: args.max_rollouts]
        comp.N_ROLLOUTS = len(init_states)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = comp.PROJECT_ROOT / "results" / f"lambda_heatmap_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===================================================")
    print("Lambda heatmap comparison")
    print(f"zeta values = {zeta_values}")
    print(f"wn values   = {wn_values}")
    print(f"lambdas     = {lambda_values}")
    print(f"run ids     = {[run_id for run_id, _, _ in runs]}")
    print(f"rollouts    = {len(init_states)}")
    print(f"output      = {out_dir}")
    print("===================================================\n")

    rows = evaluate_all(
        zeta_values=zeta_values,
        wn_values=wn_values,
        runs=runs,
        init_states=init_states,
    )

    if not rows:
        raise RuntimeError("No evaluation rows were generated.")

    diff_rows = build_diff_rows(
        rows=rows,
        zeta_values=zeta_values,
        wn_values=wn_values,
    )

    if not diff_rows:
        raise RuntimeError("No reward-difference rows were generated.")

    summary_csv = out_dir / "final_summary_metrics.csv"
    diff_csv = out_dir / "final_reward_diff_vs_lambda0.csv"

    save_rows_csv(rows, summary_csv)
    save_diff_csv(diff_rows, runs, diff_csv)

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {diff_csv}")

    if not args.no_heatmap:
        plot_heatmaps_by_zeta(
            diff_rows=diff_rows,
            lambda_values=lambda_values,
            wn_values=wn_values,
            zeta_values=zeta_values,
            out_dir=out_dir,
            annotate=not args.no_annotate_heatmap,
        )

    if args.line_plot:
        plot_line_plots_by_zeta(
            diff_rows=diff_rows,
            runs=runs,
            wn_values=wn_values,
            zeta_values=zeta_values,
            out_dir=out_dir,
            plot_std=args.plot_std,
        )


if __name__ == "__main__":
    main()