"""The scripted controller solves the task, so it is a fair reference point."""

from __future__ import annotations

from mujoco_ssl import GoToBallEnv
from mujoco_ssl.baseline import GoToBallController


def test_baseline_reaches_the_ball():
    env, controller = GoToBallEnv(), GoToBallController()
    successes = 0
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        while True:
            obs, _, terminated, truncated, info = env.step(controller(obs))
            if terminated or truncated:
                break
        successes += info["is_success"]
    env.close()
    assert successes >= 19
