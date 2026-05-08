# Second-Order Inner-Loop Drone Transfer

This repository contains code for evaluating PPO policies trained with different attitude-reference smoothness penalties and deployed on a high-order drone model with a second-order attitude inner loop.

The main goal is to compare policies trained with different lambda values:

lambda = 0, 5, 10, 15

under different second-order inner-loop parameters, especially natural frequency omega_n and damping ratio zeta.

## Model

The full internal environment state is:

s_k = [x_k, xdot_k, z_k, zdot_k, theta_k, theta_dot_k]^T

The PPO policy observes only the reduced translational state:

o_k = [x_k, xdot_k, z_k, zdot_k]^T

The policy action is:

a_k = [Delta T_k, theta_star_k]^T

where Delta T is the thrust correction and theta_star is the desired attitude reference.

The second-order attitude inner loop is:

J theta_ddot = -Kp(theta - theta_star) - Kd(theta_dot - theta_star_dot_est)

The gains are parameterized by natural frequency and damping ratio:

Kp = J omega_n^2

Kd = 2 J zeta omega_n

## Repository Structure

src/
    Environment and dynamics code.

scripts/
    Evaluation scripts.

data/
    Fixed evaluation initial states.

models/lambda_policies/
    Trained PPO policies and VecNormalize statistics.

results/
    Generated output files. This folder is ignored by Git except for .gitkeep.

docs/figures/
    Selected figures included in the repository.

## Main Scripts

### 1. Compact Summary Evaluation

This is the recommended script for normal experiments:

python scripts/evaluate_lambda_summary.py --zeta 0.4 --wn_values 2 4 6 8 10 12

It reuses the trusted comprehensive evaluation function, but avoids generating all detailed time-series plots.

It saves summary files such as:

final_summary_metrics.csv

final_reward_diff_vs_lambda0.csv

final_reward_diff_vs_lambda0_zeta_*.png

The main plotted quantity is:

J_rew(lambda) - J_rew(lambda=0)

This shows the improvement or degradation of each lambda policy relative to the lambda=0 baseline.

Example with a different damping ratio:

python scripts/evaluate_lambda_summary.py --zeta 0.7 --wn_values 2 4 6 8 10 12

Quick test with fewer initial conditions:

python scripts/evaluate_lambda_summary.py --zeta 0.7 --wn_values 2 4 --max_rollouts 20

Run without opening a plot window:

python scripts/evaluate_lambda_summary.py --zeta 0.7 --wn_values 2 4 6 8 10 12 --no_plot

### 2. Full Diagnostic Comparison

This script generates the full set of detailed plots:

python scripts/evaluate_lambda_comparison.py

Use this script when detailed diagnostics are needed, such as position error, cumulative reward, cumulative return, attitude tracking, velocity, thrust, and trajectory plots.

## Installation

Install the required packages with:

python -m pip install -r requirements.txt

## Syntax Check

To check the Python files before running:

python -m py_compile src\env_second_order_inner_loop.py scripts\evaluate_lambda_comparison.py scripts\evaluate_lambda_summary.py

## Notes

- The compact summary script is recommended for reporting and quick comparison.
- The full comparison script is useful for debugging and detailed analysis.
- Generated result folders are ignored by Git to keep the repository clean.
- Selected important figures can be copied to docs/figures/ if they should appear on GitHub.
