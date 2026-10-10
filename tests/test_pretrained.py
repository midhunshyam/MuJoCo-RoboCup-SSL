"""The shipped policy loads and solves a few episodes (needs the `train` extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mujoco_ssl import GoToBallEnv

MODEL = Path(__file__).resolve().parents[1] / "models" / "ppo_go_to_ball.zip"


def test_pretrained_policy_reaches_the_ball():
    sb3 = pytest.importorskip("stable_baselines3")
    if not MODEL.exists():
        pytest.skip("no trained model in models/")
    model = sb3.PPO.load(MODEL, device="cpu")
    env = GoToBallEnv()
    assert model.observation_space.shape == env.observation_space.shape
    assert np.allclose(model.action_space.low, env.action_space.low)
    successes = 0
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += info["is_success"]
    env.close()
    assert successes >= 9
