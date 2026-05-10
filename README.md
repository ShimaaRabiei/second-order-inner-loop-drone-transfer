# Second-Order Inner-Loop Drone Transfer

This repository contains code for training reduced-order PPO policies and evaluating their zero-shot transfer to a high-order drone model with a second-order attitude inner loop.

The main goal is to compare policies trained with different attitude-reference smoothness penalties:

$$
\lambda \in \{0, 5, 10, 15\}
$$

The trained reduced-order policies are deployed on a high-order drone model where the policy outputs a desired attitude reference, and a second-order inner-loop controller tracks this reference.

The experiments evaluate how the smoothness penalty affects transfer performance under different inner-loop parameters, especially the natural frequency `omega_n` and damping ratio `zeta`.

---

## Model Overview

### Full High-Order Drone Model

The full internal high-order state is:

$$
s_k =
[x_k, \dot{x}_k, z_k, \dot{z}_k, \theta_k, \dot{\theta}_k]^T
$$

The PPO policy does not observe the full state. It observes only the reduced translational state:

$$
o_k =
[x_k, \dot{x}_k, z_k, \dot{z}_k]^T
$$

The policy action is:

$$
a_k =
[\Delta T_k, \theta_k^\star]^T
$$

where `Delta T` is the thrust correction and `theta_star` is the desired attitude reference.

The high-order model uses a second-order attitude inner loop:

$$
J \ddot{\theta}
=
-K_p(\theta - \theta^\star)
-
K_d(\dot{\theta} - \dot{\theta}^{\star}_{\mathrm{est}})
$$

The gains are parameterized by natural frequency and damping ratio:

$$
K_p = J \omega_n^2
$$

$$
K_d = 2J\zeta\omega_n
$$

---

## Reduced-Order Training Model

The reduced-order model is used to train the PPO policy before transfer.

The reduced-order observation is:

$$
o_k =
[x_k, \dot{x}_k, z_k, \dot{z}_k]^T
$$

The reduced-order action is:

$$
a_k =
[\Delta T_k, \theta_k^\star]^T
$$

In the reduced-order model, the second action component is applied directly as the attitude reference. During transfer, this same policy output becomes the desired attitude command tracked by the high-order second-order inner loop.

The reduced-order training reward includes a distance-to-target term and an optional smoothness penalty on changes in the desired attitude reference:

$$
r_k^{\lambda}
=
r_k
-
\lambda |\theta_k^\star - \theta_{k-1}^\star|
$$

where `lambda` controls how strongly the policy is encouraged to produce smoother attitude-reference commands.

---

## Repository Structure

```text
src/
    Environment and dynamics code.

scripts/
    Training and evaluation scripts.

data/
    Fixed evaluation initial states.

models/lambda_policies/
    Trained PPO policies and VecNormalize statistics.

results/
    Generated output files. This folder is ignored by Git except for .gitkeep.

docs/figures/
    Selected figures included in the repository.
```

---

## Main Files

### Reduced-Order Training and Evaluation

`src/reduced_order_drone_env.py`

Defines the reduced-order drone environment, reward-cost wrapper, training constants, and helper function `make_env`.

`scripts/train_reduced_order_policy.py`

Trains or warm-starts a reduced-order PPO policy.

`scripts/evaluate_reduced_order_rollout.py`

Evaluates one deterministic rollout of a trained reduced-order policy.

### High-Order Transfer Evaluation

`src/env_second_order_inner_loop.py`

Defines the high-order drone model with the second-order attitude inner loop.

`scripts/evaluate_lambda_summary.py`

Runs compact transfer evaluations for multiple lambda policies and inner-loop parameters.

`scripts/evaluate_lambda_comparison.py`

Runs full diagnostic transfer evaluations and generates detailed plots.

`scripts/evaluate_single_rollout.py`

Evaluates and visualizes one high-order rollout.

---

## Installation

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

---

## Reduced-Order Policy Training

To train a reduced-order policy from scratch with `lambda = 0`, run:

```bash
python scripts\train_reduced_order_policy.py --lambda_pen 0
```

To train with another lambda value, for example `lambda = 15`, run:

```bash
python scripts\train_reduced_order_policy.py --lambda_pen 15
```

The trained policy and VecNormalize statistics are saved under:

`models/lambda_policies/`

For example:

```text
models/lambda_policies/lambda_0_policy.zip
models/lambda_policies/lambda_0_vecnormalize.pkl
```

---

## Warm-Start Reduced-Order Training

To warm-start from an existing PPO policy and VecNormalize statistics, provide the previous model and vector normalization files explicitly.

Example:

```bash
python scripts\train_reduced_order_policy.py --lambda_pen 0 --tag test1 --prev_model "C:\Users\rabiei\Drone\000\ppo_drone_fresh.zip" --prev_vecnormalize "C:\Users\rabiei\Drone\000\norm_fresh.pkl"
```

This saves the new warm-started policy as:

```text
models/lambda_policies/lambda_0_test1_policy.zip
models/lambda_policies/lambda_0_test1_vecnormalize.pkl
```

The optional `--tag` argument is useful for testing new training runs without overwriting existing policies.

---

## Reduced-Order Rollout Evaluation

To evaluate a reduced-order policy using explicit model and VecNormalize paths, run:

```bash
python scripts\evaluate_reduced_order_rollout.py --lambda_pen 0 --model "models\lambda_policies\lambda_0_test1_policy.zip" --vecnormalize "models\lambda_policies\lambda_0_test1_vecnormalize.pkl"
```

By default, the evaluation uses the fixed initial state:

$$
[x_0, \dot{x}_0, z_0, \dot{z}_0] = [1, 1, 1, 1]
$$

To use a random initial condition instead, run:

```bash
python scripts\evaluate_reduced_order_rollout.py --lambda_pen 0 --model "models\lambda_policies\lambda_0_test1_policy.zip" --vecnormalize "models\lambda_policies\lambda_0_test1_vecnormalize.pkl" --random_start
```

To specify a different fixed state, run for example:

```bash
python scripts\evaluate_reduced_order_rollout.py --lambda_pen 0 --model "models\lambda_policies\lambda_0_test1_policy.zip" --vecnormalize "models\lambda_policies\lambda_0_test1_vecnormalize.pkl" --fixed_state 1 1 1 1
```

---

## High-Order Transfer Evaluation

### Compact Summary Evaluation

This is the recommended script for normal experiments:

```bash
python scripts\evaluate_lambda_summary.py --zeta 0.4 --wn_values 2 4 6 8 10 12
```

It saves summary files such as:

```text
final_summary_metrics.csv
final_reward_diff_vs_lambda0.csv
final_reward_diff_vs_lambda0_zeta_*.png
```

The main plotted quantity is:

$$
J_{\mathrm{rew}}(\lambda) - J_{\mathrm{rew}}(\lambda=0)
$$

This shows the improvement or degradation of each lambda policy relative to the `lambda = 0` baseline.

Example with a different damping ratio:

```bash
python scripts\evaluate_lambda_summary.py --zeta 0.7 --wn_values 2 4 6 8 10 12
```

Quick test with fewer initial conditions:

```bash
python scripts\evaluate_lambda_summary.py --zeta 0.7 --wn_values 2 4 --max_rollouts 20
```

Run without opening a plot window:

```bash
python scripts\evaluate_lambda_summary.py --zeta 0.7 --wn_values 2 4 6 8 10 12 --no_plot
```

---

## Full Diagnostic Comparison

This script generates the full set of detailed plots:

```bash
python scripts\evaluate_lambda_comparison.py
```

Use this script when detailed diagnostics are needed, such as position error, cumulative reward, cumulative return, attitude tracking, velocity, thrust, and trajectory plots.

---

## Syntax Check

To check the main Python files before running:

```bash
python -m py_compile src\env_second_order_inner_loop.py src\reduced_order_drone_env.py scripts\train_reduced_order_policy.py scripts\evaluate_reduced_order_rollout.py scripts\evaluate_lambda_comparison.py scripts\evaluate_lambda_summary.py
```

---

## Notes

- The reduced-order training script can train from scratch or warm-start from a previous PPO model and VecNormalize file.
- The reduced-order evaluation script is useful for checking whether the trained policy behaves well before testing transfer on the high-order model.
- The compact summary script is recommended for reporting and quick comparison of high-order transfer performance.
- The full comparison script is useful for debugging and detailed trajectory-level analysis.
- Generated result folders are ignored by Git to keep the repository clean.
- Selected important figures can be copied to `docs/figures/` if they should appear on GitHub.
- Test models such as `lambda_0_test1_policy.zip` and `lambda_0_test1_vecnormalize.pkl` can be kept for reproducibility if the files are small enough for GitHub.