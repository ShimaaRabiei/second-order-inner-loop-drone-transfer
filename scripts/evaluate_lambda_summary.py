# -*- coding: utf-8 -*-
"""
Compact lambda-comparison evaluator.

This script reuses the comprehensive evaluator in
evaluate_lambda_comparison.py, but avoids generating all per-time plots.

It evaluates selected lambda policies over selected omega_n values and a selected
zeta, then saves:
    - final_summary_metrics.csv
    - final_reward_diff_vs_lambda0.csv
    - optional one summary plot
    - optional std/error bars in the plot

Examples:
    python scripts/evaluate_lambda_summary.py --zeta 0.7 --wn_values 2 4 6 8 10 12 --lambdas 0 4 4.8 5 5.2 10 15

    python scripts/evaluate_lambda_summary.py --zeta 0.7 --wn_values 2 4 6 8 10 12 --lambdas 0 4 4.8 5 5.2 10 15 --plot_std

    python scripts/evaluate_lambda_summary.py --zeta 0.7 --wn_values 4 8 12 --lambdas 0 4.8 5 5.2 --max_rollouts 10 --plot_std
"""

import argparse
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Import trusted comprehensive comparison code
import evaluate_lambda_comparison as comp


def lambda_to_run_id(lam_val):
    """
    Convert a lambda value to the saved model/run folder name.

    Examples:
        0.0  -> lambda_0
        4.0  -> lambda_4
        4.8  -> lambda_4.8
        5.2  -> lambda_5.2

    If your folders are named differently, change only this function.
    """
    return f"lambda_{lam_val:g}"


def lambda_to_label(lam_val):
    """
    Convert lambda value to a plot label.
    """
    return rf"$\lambda={lam_val:g}$"


def unique_lambdas(lambdas):
    """
    Remove duplicate lambda values while preserving order.
    """
    result = []

    for lam in lambdas:
        lam = float(lam)

        if not any(abs(lam - existing) < 1e-12 for existing in result):
            result.append(lam)

    return result


def last_valid_value(arr):
    arr = np.asarray(arr, dtype=float)
    valid = ~np.isnan(arr)

    if not np.any(valid):
        return np.nan

    return float(arr[valid][-1])


def last_valid_metric(metrics, key):
    """
    Safely get the last valid value of metrics[key].
    """
    if key not in metrics:
        return np.nan

    return last_valid_value(metrics[key])


def last_valid_std(metrics, mean_key):
    """
    Safely get the last valid std corresponding to a mean key.

    Example:
        cum_rew_mean -> cum_rew_std

    If the comprehensive evaluator does not return std arrays, this returns NaN.
    """
    std_key = mean_key.replace("_mean", "_std")

    if std_key not in metrics:
        return np.nan

    return last_valid_value(metrics[std_key])


def combine_std_for_difference(std_a, std_b):
    """
    Approximate std of A - B.

    This assumes independent quantities:
        std(A - B) approx sqrt(std(A)^2 + std(B)^2)

    Since all policies are usually evaluated on the same initial states,
    the true paired std may be smaller. This is still a useful conservative
    visual summary when only aggregate std arrays are available.
    """
    if np.isnan(std_a) or np.isnan(std_b):
        return np.nan

    return float(np.sqrt(std_a**2 + std_b**2))


def main():
    parser = argparse.ArgumentParser(
        description="Run compact lambda comparison using the trusted comprehensive evaluator."
    )

    parser.add_argument(
        "--zeta",
        type=float,
        default=0.7,
        help="Damping ratio zeta used in the second-order inner loop.",
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
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 4.8, 5.0, 5.2, 10.0, 15.0],
        help=(
            "List of lambda values to evaluate. "
            "The script automatically looks for runs named lambda_<value>, "
            "for example lambda_4.8 and lambda_5.2."
        ),
    )

    parser.add_argument(
        "--max_rollouts",
        type=int,
        default=None,
        help="Optional limit on number of initial states. Useful for quick tests.",
    )

    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Do not create the summary plot; save CSV files only.",
    )

    parser.add_argument(
        "--plot_std",
        action="store_true",
        help="Add approximate std error bars to the summary plot.",
    )

    args = parser.parse_args()

    zeta_eval = float(args.zeta)
    wn_values = [float(v) for v in args.wn_values]

    lambda_values = unique_lambdas(args.lambdas)

    # Because the summary plot and CSV compare everything relative to lambda=0,
    # automatically include lambda=0 if the user forgot it.
    if not any(abs(lam - 0.0) < 1e-12 for lam in lambda_values):
        print("lambda=0 was not provided, so it is added automatically as the baseline.")
        lambda_values = [0.0] + lambda_values

    runs = [
        (lambda_to_run_id(lam_val), lam_val, lambda_to_label(lam_val))
        for lam_val in lambda_values
    ]

    # Optional fast/debug mode: use only first max_rollouts initial states.
    init_states = comp.INIT_STATES

    if args.max_rollouts is not None:
        init_states = init_states[: args.max_rollouts]
        comp.N_ROLLOUTS = len(init_states)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        comp.PROJECT_ROOT
        / "results"
        / f"lambda_summary_zeta_{zeta_eval:g}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n===================================================")
    print("Compact lambda comparison")
    print(f"zeta      = {zeta_eval}")
    print(f"wn values = {wn_values}")
    print(f"rollouts  = {len(init_states)}")
    print(f"lambdas   = {lambda_values}")
    print(f"run ids   = {[run_id for run_id, _, _ in runs]}")
    print(f"plot std  = {args.plot_std}")
    print(f"output    = {out_dir}")
    print("===================================================\n")

    for wn_val in wn_values:
        print(f"\n===== Evaluating omega_n = {wn_val:g}, zeta = {zeta_eval:g} =====")

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

                "final_cum_discounted_reward_mean": last_valid_metric(metrics, "cum_rew_mean"),
                "final_cum_discounted_reward_std": last_valid_std(metrics, "cum_rew_mean"),

                "final_cum_discounted_return_mean": last_valid_metric(metrics, "cum_ret_mean"),
                "final_cum_discounted_return_std": last_valid_std(metrics, "cum_ret_mean"),

                "final_position_error_mean": last_valid_metric(metrics, "pos_err_mean"),
                "final_position_error_std": last_valid_std(metrics, "pos_err_mean"),

                "final_tracking_pre_mean": last_valid_metric(metrics, "cum_tracking_pre_mean"),
                "final_tracking_pre_std": last_valid_std(metrics, "cum_tracking_pre_mean"),

                "final_tracking_post_mean": last_valid_metric(metrics, "cum_tracking_post_mean"),
                "final_tracking_post_std": last_valid_std(metrics, "cum_tracking_post_mean"),

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

    if not rows:
        raise RuntimeError("No evaluation rows were generated.")

    # ------------------------------------------------------------
    # Save detailed long-format summary
    # ------------------------------------------------------------
    summary_csv = out_dir / "final_summary_metrics.csv"
    headers = list(rows[0].keys())

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------
    # Build reward difference relative to lambda=0
    # ------------------------------------------------------------
    diff_rows = []

    for wn_val in wn_values:
        base_mean = None
        base_std = None

        for row in rows:
            if abs(row["wn"] - wn_val) < 1e-12 and abs(row["lambda"]) < 1e-12:
                base_mean = row["final_cum_discounted_reward_mean"]
                base_std = row["final_cum_discounted_reward_std"]
                break

        if base_mean is None:
            print(f"Warning: no lambda=0 baseline found for omega_n={wn_val:g}. Skipping.")
            continue

        diff_row = {
            "zeta": zeta_eval,
            "wn": wn_val,
        }

        for row in rows:
            if abs(row["wn"] - wn_val) < 1e-12:
                lam = row["lambda"]

                mean_key = f"reward_lambda_{lam:g}_minus_lambda_0_mean"
                std_key = f"reward_lambda_{lam:g}_minus_lambda_0_std_approx"

                diff_row[mean_key] = row["final_cum_discounted_reward_mean"] - base_mean

                if abs(lam) < 1e-12:
                    diff_row[std_key] = 0.0
                else:
                    diff_row[std_key] = combine_std_for_difference(
                        row["final_cum_discounted_reward_std"],
                        base_std,
                    )

        diff_rows.append(diff_row)

    if not diff_rows:
        raise RuntimeError("No reward-difference rows were generated.")

    diff_csv = out_dir / "final_reward_diff_vs_lambda0.csv"

    diff_headers = ["zeta", "wn"]

    for _, lam_val, _ in runs:
        mean_key = f"reward_lambda_{lam_val:g}_minus_lambda_0_mean"
        std_key = f"reward_lambda_{lam_val:g}_minus_lambda_0_std_approx"

        if mean_key not in diff_headers:
            diff_headers.append(mean_key)

        if std_key not in diff_headers:
            diff_headers.append(std_key)

    with open(diff_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diff_headers)
        writer.writeheader()
        writer.writerows(diff_rows)

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {diff_csv}")

    # ------------------------------------------------------------
    # Optional compact summary plot
    # final cumulative discounted reward(lambda) - reward(lambda=0)
    # ------------------------------------------------------------
    if not args.no_plot:
        plt.figure(figsize=(10, 5.5))

        # Plot lambda=0 baseline as a zero reference curve
        xs0 = [row["wn"] for row in diff_rows]
        ys0 = [0.0 for _ in diff_rows]

        plt.plot(
            xs0,
            ys0,
            linestyle=":",
            marker="o",
            label=r"$\lambda=0$ baseline",
        )

        # Plot reward(lambda) - reward(lambda=0) for lambda > 0
        for _, lam_val, lam_label in runs:
            if abs(lam_val) < 1e-12:
                continue

            mean_key = f"reward_lambda_{lam_val:g}_minus_lambda_0_mean"
            std_key = f"reward_lambda_{lam_val:g}_minus_lambda_0_std_approx"

            xs = []
            ys = []
            yerrs = []

            for row in diff_rows:
                if mean_key in row:
                    xs.append(row["wn"])
                    ys.append(row[mean_key])
                    yerrs.append(row.get(std_key, np.nan))

            if len(xs) > 0:
                if args.plot_std and not np.all(np.isnan(yerrs)):
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
            rf"for different $\omega_n$, $\zeta={zeta_eval:g}$"
        )

        plt.legend()
        plt.tight_layout()

        plot_path = out_dir / f"final_reward_diff_vs_lambda0_zeta_{zeta_eval:g}.png"
        plt.savefig(plot_path, dpi=300)
        plt.show()

        print(f"  {plot_path}")


if __name__ == "__main__":
    main()