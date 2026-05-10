# \# Second-Order Inner-Loop Drone Transfer

# 

# This repository contains code for training reduced-order PPO policies and evaluating their zero-shot transfer to a high-order drone model with a second-order attitude inner loop.

# 

# The main goal is to compare policies trained with different attitude-reference smoothness penalties:

# 

# ```math

# \\lambda \\in \\{0, 5, 10, 15\\}

# ```

# 

# The trained reduced-order policies are deployed on a high-order drone model where the policy outputs a desired attitude reference, and a second-order inner-loop controller tracks this reference.

# 

# The experiments evaluate how the smoothness penalty affects transfer performance under different inner-loop parameters, especially the natural frequency `omega\_n` and damping ratio `zeta`.

# 

# \---

# 

# \## Model Overview

# 

# \### Full High-Order Drone Model

# 

# The full internal high-order state is:

# 

# ```math

# s\_k =

# \[x\_k, \\dot{x}\_k, z\_k, \\dot{z}\_k, \\theta\_k, \\dot{\\theta}\_k]^T

# ```

# 

# The PPO policy does not observe the full state. It observes only the reduced translational state:

# 

# ```math

# o\_k =

# \[x\_k, \\dot{x}\_k, z\_k, \\dot{z}\_k]^T

# ```

# 

# The policy action is:

# 

# ```math

# a\_k =

# \[\\Delta T\_k, \\theta\_k^\\star]^T

# ```

# 

# where `Delta T` is the thrust correction and `theta\_star` is the desired attitude reference.

# 

# The high-order model uses a second-order attitude inner loop:

# 

# ```math

# J \\ddot{\\theta}

# =

# \-K\_p(\\theta - \\theta^\\star)

# \-

# K\_d(\\dot{\\theta} - \\dot{\\theta}^{\\star}\_{\\mathrm{est}})

# ```

# 

# The gains are parameterized by natural frequency and damping ratio:

# 

# ```math

# K\_p = J \\omega\_n^2

# ```

# 

# ```math

# K\_d = 2J\\zeta\\omega\_n

# ```

# 

# \---

# 

# \## Reduced-Order Training Model

# 

# The reduced-order model is used to train the PPO policy before transfer.

# 

# The reduced-order observation is:

# 

# ```math

# o\_k =

# \[x\_k, \\dot{x}\_k, z\_k, \\dot{z}\_k]^T

# ```

# 

# The reduced-order action is:

# 

# ```math

# a\_k =

# \[\\Delta T\_k, \\theta\_k^\\star]^T

# ```

# 

# In the reduced-order model, the second action component is applied directly as the attitude reference. During transfer, this same policy output becomes the desired attitude command tracked by the high-order second-order inner loop.

# 

# The reduced-order training reward includes a distance-to-target term and an optional smoothness penalty on changes in the desired attitude reference:

# 

# ```math

# r\_k^{\\lambda}

# =

# r\_k

# \-

# \\lambda |\\theta\_k^\\star - \\theta\_{k-1}^\\star|

# ```

# 

# where `lambda` controls how strongly the policy is encouraged to produce smoother attitude-reference commands.

# 

# \---

# 

# \## Repository Structure

# 

# ```text

# src/

# &#x20;   Environment and dynamics code.

# 

# scripts/

# &#x20;   Training and evaluation scripts.

# 

# data/

# &#x20;   Fixed evaluation initial states.

# 

# models/lambda\_policies/

# &#x20;   Trained PPO policies and VecNormalize statistics.

# 

# results/

# &#x20;   Generated output files. This folder is ignored by Git except for .gitkeep.

# 

# docs/figures/

# &#x20;   Selected figures included in the repository.

# ```

# 

# \---

# 

# \## Main Files

# 

# \### Reduced-Order Training and Evaluation

# 

# ```text

# src/reduced\_order\_drone\_env.py

# ```

# 

# Defines the reduced-order drone environment, reward-cost wrapper, training constants, and helper function `make\_env`.

# 

# ```text

# scripts/train\_reduced\_order\_policy.py

# ```

# 

# Trains or warm-starts a reduced-order PPO policy.

# 

# ```text

# scripts/evaluate\_reduced\_order\_rollout.py

# ```

# 

# Evaluates one deterministic rollout of a trained reduced-order policy.

# 

# \### High-Order Transfer Evaluation

# 

# ```text

# src/env\_second\_order\_inner\_loop.py

# ```

# 

# Defines the high-order drone model with the second-order attitude inner loop.

# 

# ```text

# scripts/evaluate\_lambda\_summary.py

# ```

# 

# Runs compact transfer evaluations for multiple lambda policies and inner-loop parameters.

# 

# ```text

# scripts/evaluate\_lambda\_comparison.py

# ```

# 

# Runs full diagnostic transfer evaluations and generates detailed plots.

# 

# ```text

# scripts/evaluate\_single\_rollout.py

# ```

# 

# Evaluates and visualizes one high-order rollout.

# 

# \---

# 

# \## Installation

# 

# Install the required packages with:

# 

# ```bash

# python -m pip install -r requirements.txt

# ```

# 

# \---

# 

# \## Reduced-Order Policy Training

# 

# To train a reduced-order policy from scratch with `lambda = 0`, run:

# 

# ```bash

# python scripts\\train\_reduced\_order\_policy.py --lambda\_pen 0

# ```

# 

# To train with another lambda value, for example `lambda = 15`, run:

# 

# ```bash

# python scripts\\train\_reduced\_order\_policy.py --lambda\_pen 15

# ```

# 

# The trained policy and VecNormalize statistics are saved under:

# 

# ```text

# models/lambda\_policies/

# ```

# 

# For example:

# 

# ```text

# models/lambda\_policies/lambda\_0\_policy.zip

# models/lambda\_policies/lambda\_0\_vecnormalize.pkl

# ```

# 

# \---

# 

# \## Warm-Start Reduced-Order Training

# 

# To warm-start from an existing PPO policy and VecNormalize statistics, provide the previous model and vector normalization files explicitly.

# 

# Example:

# 

# ```bash

# python scripts\\train\_reduced\_order\_policy.py --lambda\_pen 0 --tag test1 --prev\_model "C:\\Users\\rabiei\\Drone\\000\\ppo\_drone\_fresh.zip" --prev\_vecnormalize "C:\\Users\\rabiei\\Drone\\000\\norm\_fresh.pkl"

# ```

# 

# This saves the new warm-started policy as:

# 

# ```text

# models/lambda\_policies/lambda\_0\_test1\_policy.zip

# models/lambda\_policies/lambda\_0\_test1\_vecnormalize.pkl

# ```

# 

# The optional `--tag` argument is useful for testing new training runs without overwriting existing policies.

# 

# \---

# 

# \## Reduced-Order Rollout Evaluation

# 

# To evaluate a reduced-order policy using explicit model and VecNormalize paths, run:

# 

# ```bash

# python scripts\\evaluate\_reduced\_order\_rollout.py --lambda\_pen 0 --model "models\\lambda\_policies\\lambda\_0\_test1\_policy.zip" --vecnormalize "models\\lambda\_policies\\lambda\_0\_test1\_vecnormalize.pkl"

# ```

# 

# By default, the evaluation uses the fixed initial state:

# 

# ```math

# \[x\_0, \\dot{x}\_0, z\_0, \\dot{z}\_0] = \[1, 1, 1, 1]

# ```

# 

# To use a random initial condition instead, run:

# 

# ```bash

# python scripts\\evaluate\_reduced\_order\_rollout.py --lambda\_pen 0 --model "models\\lambda\_policies\\lambda\_0\_test1\_policy.zip" --vecnormalize "models\\lambda\_policies\\lambda\_0\_test1\_vecnormalize.pkl" --random\_start

# ```

# 

# To specify a different fixed state, run for example:

# 

# ```bash

# python scripts\\evaluate\_reduced\_order\_rollout.py --lambda\_pen 0 --model "models\\lambda\_policies\\lambda\_0\_test1\_policy.zip" --vecnormalize "models\\lambda\_policies\\lambda\_0\_test1\_vecnormalize.pkl" --fixed\_state 1 1 1 1

# ```

# 

# \---

# 

# \## High-Order Transfer Evaluation

# 

# \### Compact Summary Evaluation

# 

# This is the recommended script for normal experiments:

# 

# ```bash

# python scripts\\evaluate\_lambda\_summary.py --zeta 0.4 --wn\_values 2 4 6 8 10 12

# ```

# 

# It saves summary files such as:

# 

# ```text

# final\_summary\_metrics.csv

# final\_reward\_diff\_vs\_lambda0.csv

# final\_reward\_diff\_vs\_lambda0\_zeta\_\*.png

# ```

# 

# The main plotted quantity is:

# 

# ```math

# J\_{\\mathrm{rew}}(\\lambda) - J\_{\\mathrm{rew}}(\\lambda=0)

# ```

# 

# This shows the improvement or degradation of each lambda policy relative to the `lambda = 0` baseline.

# 

# Example with a different damping ratio:

# 

# ```bash

# python scripts\\evaluate\_lambda\_summary.py --zeta 0.7 --wn\_values 2 4 6 8 10 12

# ```

# 

# Quick test with fewer initial conditions:

# 

# ```bash

# python scripts\\evaluate\_lambda\_summary.py --zeta 0.7 --wn\_values 2 4 --max\_rollouts 20

# ```

# 

# Run without opening a plot window:

# 

# ```bash

# python scripts\\evaluate\_lambda\_summary.py --zeta 0.7 --wn\_values 2 4 6 8 10 12 --no\_plot

# ```

# 

# \---

# 

# \## Full Diagnostic Comparison

# 

# This script generates the full set of detailed plots:

# 

# ```bash

# python scripts\\evaluate\_lambda\_comparison.py

# ```

# 

# Use this script when detailed diagnostics are needed, such as position error, cumulative reward, cumulative return, attitude tracking, velocity, thrust, and trajectory plots.

# 

# \---

# 

# \## Syntax Check

# 

# To check the main Python files before running:

# 

# ```bash

# python -m py\_compile src\\env\_second\_order\_inner\_loop.py src\\reduced\_order\_drone\_env.py scripts\\train\_reduced\_order\_policy.py scripts\\evaluate\_reduced\_order\_rollout.py scripts\\evaluate\_lambda\_comparison.py scripts\\evaluate\_lambda\_summary.py

# ```

# 

# \---

# 

# \## Notes

# 

# \- The reduced-order training script can train from scratch or warm-start from a previous PPO model and VecNormalize file.

# \- The reduced-order evaluation script is useful for checking whether the trained policy behaves well before testing transfer on the high-order model.

# \- The compact summary script is recommended for reporting and quick comparison of high-order transfer performance.

# \- The full comparison script is useful for debugging and detailed trajectory-level analysis.

# \- Generated result folders are ignored by Git to keep the repository clean.

# \- Selected important figures can be copied to `docs/figures/` if they should appear on GitHub.

# \- Test models such as `lambda\_0\_test1\_policy.zip` and `lambda\_0\_test1\_vecnormalize.pkl` can be kept for reproducibility if the files are small enough for GitHub.

