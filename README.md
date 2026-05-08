# Second-Order Inner-Loop Drone Transfer

This repository contains code for evaluating PPO policies trained on a reduced-order drone model and deployed on a high-order drone model with a second-order attitude inner loop.

## Model

The full internal environment state is:

s_k = [x_k, xdot_k, z_k, zdot_k, theta_k, theta_dot_k]^T

The PPO policy observes only:

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

src/        Environment and dynamics code
scripts/    Evaluation scripts
data/       Fixed evaluation initial states
models/     Trained PPO policies and VecNormalize statistics
results/    Generated plots and summary files

## Installation

pip install -r requirements.txt

## Run Lambda Comparison

python scripts/evaluate_lambda_comparison.py

## Run Single-Rollout Evaluation

python scripts/evaluate_single_rollout.py

## Policies

The repository includes four trained policies corresponding to lambda values 0, 5, 10, and 15. Each policy has a corresponding VecNormalize statistics file.
