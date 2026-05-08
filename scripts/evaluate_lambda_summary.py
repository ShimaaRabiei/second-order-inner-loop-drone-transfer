# -*- coding: utf-8 -*-
"""
Compact lambda-comparison evaluator.

This script reuses the trusted comprehensive evaluator in
evaluate_lambda_comparison.py, but avoids generating all per-time plots.

It evaluates all lambda policies over selected omega_n values and a selected
zeta, then saves:
    - final_summary_metrics.csv
    - final_reward_diff_vs_lambda0.csv
    - optional one summary plot

Example:
    python scripts/evaluate_lambda_summary.py --zeta 0.4 --wn_values 2 4 6 8 10 12
    python scripts/evaluate_lambda_summary.py --zeta 0.7 --wn_values 4 8 12
    python scripts/evaluate_lambda_summary.py --zeta 0.4 --max_rollouts 30
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Import trusted comprehensive comparison code
import evaluate_lambda_comparison as comp


def last_valid_value(arr):
    arr = np.asarray(arr, dtype=float)
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return np.nan
    return float(arr[valid][-1])


def main():
    parser = argparse.ArgumentParser(
        description="Run compact lambda comparison using the trusted comprehensive evaluator."
    )

    parser.add_argument(
        "--zeta",
        type=float,
        default=0.4,
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

    args = parser.parse_args()

    zeta_eval = float(args.zeta)
    wn_values = [float(v) for v in args.wn_values]

    # Optional fast/debug mode: use only first max_rollouts initial states.
    init_states = comp.INIT_STATES
    if args.max_rollouts is not None:
        init_states = init_states[: args.max_rollouts]
        comp.N_ROLLOUTS = len(init_states)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = comp.PROJECT_ROOT / "results" / f"lambda_summary_zeta_{zeta_eval:g}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n===================================================")
    print("Compact lambda comparison")
    print(f"zeta      = {zeta_eval}")
    print(f"wn values = {wn_values}")
    print(f"rollouts  = {len(init_states)}")
    print(f"output    = {out_dir}")
    print("===================================================\n")

    for wn_val in wn_values:
        print(f"\n===== Evaluating omega_n = {wn_val:g}, zeta = {zeta_eval:g} =====")

        for run_id, lam_val, lam_label in comp.RUNS:
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
                "final_cum_discounted_reward": last_valid_value(metrics["cum_rew_mean"]),
                "final_cum_discounted_return": last_valid_value(metrics["cum_ret_mean"]),
                "final_position_error": last_valid_value(metrics["pos_err_mean"]),
                "final_tracking_pre": last_valid_value(metrics["cum_tracking_pre_mean"]),
                "final_tracking_post": last_valid_value(metrics["cum_tracking_post_mean"]),
                "final_x": last_valid_value(metrics["x_mean"]),
                "final_z": last_valid_value(metrics["z_mean"]),
                "final_vx": last_valid_value(metrics["vx_mean"]),
                "final_vz": last_valid_value(metrics["vz_mean"]),
            }
            rows.append(row)

            print(
                f"lambda={lam_val:>5g} | "
                f"reward={row['final_cum_discounted_reward']:+.4f} | "
                f"return={row['final_cum_discounted_return']:+.4f} | "
                f"pos_err={row['final_position_error']:.4f}"
            )

    # Save detailed long-format summary
    summary_csv = out_dir / "final_summary_metrics.csv"
    headers = list(rows[0].keys())

    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    # Build reward difference vs lambda=0
    diff_rows = []
    for wn_val in wn_values:
        base = None
        for row in rows:
            if abs(row["wn"] - wn_val) < 1e-12 and abs(row["lambda"]) < 1e-12:
                base = row["final_cum_discounted_reward"]
                break

        if base is None:
            continue

        diff_row = {"zeta": zeta_eval, "wn": wn_val}
        for row in rows:
            if abs(row["wn"] - wn_val) < 1e-12:
                lam = row["lambda"]
                diff_row[f"reward_lambda_{lam:g}_minus_lambda_0"] = (
                    row["final_cum_discounted_reward"] - base
                )
        diff_rows.append(diff_row)

    diff_csv = out_dir / "final_reward_diff_vs_lambda0.csv"
    diff_headers = list(diff_rows[0].keys())

    with open(diff_csv, "w", encoding="utf-8") as f:
        f.write(",".join(diff_headers) + "\n")
        for row in diff_rows:
            f.write(",".join(str(row.get(h, "")) for h in diff_headers) + "\n")

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {diff_csv}")

    # Optional one compact summary plot
    # This plot matches the comprehensive comparison script:
    # final cumulative discounted reward(lambda) - final cumulative discounted reward(lambda=0)
    if not args.no_plot:
        plt.figure(figsize=(9, 5))

        # Plot lambda=0 baseline as a zero reference curve
        xs0 = [row["wn"] for row in diff_rows]
        ys0 = [0.0 for _ in diff_rows]
        plt.plot(xs0, ys0, linestyle=":", marker="o", label=r"$\lambda=0$ baseline")

        # Plot reward(lambda) - reward(lambda=0) for lambda > 0
        for _, lam_val, lam_label in comp.RUNS:
            if abs(lam_val) < 1e-12:
                continue

            key = f"reward_lambda_{lam_val:g}_minus_lambda_0"

            xs = []
            ys = []
            for row in diff_rows:
                xs.append(row["wn"])
                ys.append(row[key])

            plt.plot(xs, ys, marker="o", label=lam_label + r" $-$ $\lambda=0$")

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
