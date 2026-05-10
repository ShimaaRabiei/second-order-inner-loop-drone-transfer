# -*- coding: utf-8 -*-
"""Train or fine-tune a reduced-order drone PPO policy."""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "models" / "lambda_policies"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reduced_order_drone_env import (  # noqa: E402
    BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    N_ENVS,
    N_STEPS,
    SEED,
    TOTAL_STEPS,
    ThreePanelLogger,
    make_env,
)


def lambda_tag(lam: float) -> str:
    return f"lambda_{lam:g}"


def main():
    parser = argparse.ArgumentParser(description="Train/fine-tune reduced-order drone PPO policy.")
    parser.add_argument("--lambda_pen", type=float, default=0.0, help="Penalty weight on |Delta theta_star|.")
    parser.add_argument("--timesteps", type=int, default=TOTAL_STEPS, help="Training/fine-tuning timesteps.")
    parser.add_argument("--prev_model", type=str, default="", help="Optional warm-start PPO .zip path.")
    parser.add_argument("--prev_vecnormalize", type=str, default="", help="Optional warm-start VecNormalize .pkl path.")
    parser.add_argument("--new_lr", type=float, default=3e-6, help="Learning rate used for warm-start fine-tuning.")
    parser.add_argument("--new_clip", type=float, default=0.10, help="Clip range used for warm-start fine-tuning.")
    parser.add_argument("--tag", type=str, default="", help="Optional suffix added to output filenames.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    lam = float(args.lambda_pen)
    tag = lambda_tag(lam)
    if args.tag:
        tag = f"{tag}_{args.tag}"

    vec_env = SubprocVecEnv([make_env(i, lam=lam) for i in range(N_ENVS)])

    prev_vec = Path(args.prev_vecnormalize) if args.prev_vecnormalize else None
    if prev_vec and prev_vec.is_file():
        vec_env = VecNormalize.load(str(prev_vec), vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        print(f">> Loaded VecNormalize stats from {prev_vec}")
    else:
        vec_env = VecNormalize(VecMonitor(vec_env), norm_obs=True, norm_reward=True, gamma=GAMMA)
        print(">> Created fresh VecNormalize stats.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prev_model = Path(args.prev_model) if args.prev_model else None
    if prev_model and prev_model.is_file():
        model = PPO.load(str(prev_model), env=vec_env, device=device)
        model.learning_rate = get_schedule_fn(args.new_lr)
        model.clip_range = get_schedule_fn(args.new_clip)
        print(f">> Loaded PPO model from {prev_model} and continuing training.")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=0.98,
            clip_range=0.10,
            vf_coef=0.8,
            policy_kwargs=dict(
                activation_fn=torch.nn.Tanh,
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            ),
            device=device,
            verbose=1,
            seed=SEED,
        )
        print(">> Created new PPO model from scratch.")

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
        callback=ThreePanelLogger(gamma=GAMMA, lam=lam),
    )

    out_model = MODEL_DIR / f"{tag}_policy.zip"
    out_vecnormalize = MODEL_DIR / f"{tag}_vecnormalize.pkl"
    model.save(str(out_model))
    vec_env.save(str(out_vecnormalize))
    print(f"\n==> saved {out_model}")
    print(f"==> saved {out_vecnormalize}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
