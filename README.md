# Second-Order Inner-Loop Drone Transfer

This repository contains code for evaluating PPO policies trained with different attitude-reference smoothness penalties and deployed on a high-order drone model with a second-order attitude inner loop.

The main goal is to compare policies trained with different lambda values:

lambda = 0, 5, 10, 15

under different second-order inner-loop parameters, especially natural frequency omega\_n and damping ratio zeta.

## Model

The full internal environment state is:

s\_k = \[x\_k, xdot\_k, z\_k, zdot\_k, theta\_k, theta\_dot\_k]^T

The PPO policy observes only the reduced translational state:

o\_k = \[x\_k, xdot\_k, z\_k, zdot\_k]^T

The policy action is:

a\_k = \[Delta T\_k, theta\_star\_k]^T

where Delta T is the thrust correction and theta\_star is the desired attitude reference.

The second-order attitude inner loop is:

J theta\_ddot = -Kp(theta - theta\_star) - Kd(theta\_dot - theta\_star\_dot\_est)

The gains are parameterized by natural frequency and damping ratio:

Kp = J omega\_n^2

Kd = 2 J zeta omega\_n

## Repository Structure

src/
Environment and dynamics code.

scripts/
Evaluation scripts.

data/
Fixed evaluation initial states.

models/lambda\_policies/
Trained PPO policies and VecNormalize statistics.

results/
Generated output files. This folder is ignored by Git except for .gitkeep.

docs/figures/
Selected figures included in the repository.

## Main Scripts

### 1\. Compact Summary Evaluation

This is the recommended script for normal experiments:

python scripts/evaluate\_lambda\_summary.py --zeta 0.4 --wn\_values 2 4 6 8 10 12

It saves summary files such as:

final\_summary\_metrics.csv

final\_reward\_diff\_vs\_lambda0.csv

final\_reward\_diff\_vs\_lambda0\_zeta\_\*.png

The main plotted quantity is:

J\_rew(lambda) - J\_rew(lambda=0)

This shows the improvement or degradation of each lambda policy relative to the lambda=0 baseline.

Example with a different damping ratio:

python scripts/evaluate\_lambda\_summary.py --zeta 0.7 --wn\_values 2 4 6 8 10 12

Quick test with fewer initial conditions:

python scripts/evaluate\_lambda\_summary.py --zeta 0.7 --wn\_values 2 4 --max\_rollouts 20

Run without opening a plot window:

python scripts/evaluate\_lambda\_summary.py --zeta 0.7 --wn\_values 2 4 6 8 10 12 --no\_plot

### 2\. Full Diagnostic Comparison

This script generates the full set of detailed plots:

python scripts/evaluate\_lambda\_comparison.py

Use this script when detailed diagnostics are needed, such as position error, cumulative reward, cumulative return, attitude tracking, velocity, thrust, and trajectory plots.

## Installation

Install the required packages with:

python -m pip install -r requirements.txt

## Syntax Check

To check the Python files before running:

python -m py\_compile src\\env\_second\_order\_inner\_loop.py scripts\\evaluate\_lambda\_comparison.py scripts\\evaluate\_lambda\_summary.py

## Notes

* The compact summary script is recommended for reporting and quick comparison.
* The full comparison script is useful for debugging and detailed analysis.
* Generated result folders are ignored by Git to keep the repository clean.
* Selected important figures can be copied to docs/figures/ if they should appear on GitHub.

