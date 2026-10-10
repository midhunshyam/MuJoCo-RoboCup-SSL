"""Train PPO on RoboCupSSL/GoToBall-v0 with several environments in parallel.

Writes the policy (models/ppo_go_to_ball.zip by default), one Monitor CSV per environment
under --log-dir, and a JSON summary with the settings, library versions and wall-clock time.

    python scripts/train.py                      # full run
    python scripts/train.py --timesteps 300000   # quick check
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv

from mujoco_ssl import GO_TO_BALL_ID, GoToBallEnv

LEARNING_RATE = (3e-4, 0.0)  # decays linearly from the first value to the second

PPO_SETTINGS = {
    "n_steps": 1024,  # steps per environment between updates
    "batch_size": 1024,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--model-out", type=Path, default=Path("models/ppo_go_to_ball.zip"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo_go_to_ball"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/training_run.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    monitor_dir = args.log_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        GoToBallEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=str(monitor_dir),
        monitor_kwargs={"info_keywords": ("is_success",)},
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LinearSchedule(*LEARNING_RATE, end_fraction=1.0),
        seed=args.seed,
        device="cpu",
        verbose=1,
        **PPO_SETTINGS,
    )

    start = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    seconds = time.perf_counter() - start
    env.close()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)

    summary = {
        "env_id": GO_TO_BALL_ID,
        "algorithm": "PPO",
        "timesteps": int(model.num_timesteps),
        "n_envs": args.n_envs,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "wall_clock_seconds": round(seconds, 1),
        "steps_per_second": round(model.num_timesteps / seconds),
        "ppo_settings": {
            **PPO_SETTINGS,
            "learning_rate": f"linear decay from {LEARNING_RATE[0]} to {LEARNING_RATE[1]}",
        },
        "model_file": args.model_out.as_posix(),
        "machine": {
            "system": platform.system(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"trained {model.num_timesteps} steps in {seconds / 60:.1f} min -> {args.model_out}")


if __name__ == "__main__":
    main()
