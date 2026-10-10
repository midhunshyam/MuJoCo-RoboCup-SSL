"""GoToBall environment: API, seeding, success, reward and episode ends."""

from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from mujoco_ssl import BALL, FIELD, GO_TO_BALL_ID, ROBOT, GoToBallEnv

CONTACT = ROBOT.front_distance + BALL.radius  # ball centre distance when touching the front


@pytest.fixture
def env() -> GoToBallEnv:
    environment = GoToBallEnv()
    yield environment
    environment.close()


def test_passes_the_gymnasium_env_checker():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(GoToBallEnv(), skip_render_check=True)
    assert [str(w.message) for w in caught] == []


def test_registered_id_makes_the_env():
    environment = gym.make(GO_TO_BALL_ID)
    obs, info = environment.reset(seed=3)
    assert environment.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = environment.step(environment.action_space.sample())
    assert isinstance(reward, float)
    environment.close()


def test_reset_puts_the_bodies_where_requested(env):
    env.reset(seed=0, options={"robot_pose": (1.2, -0.7, 0.3), "ball_position": (2.0, 1.0)})
    assert env.data.body("robot").xpos[:2] == pytest.approx([1.2, -0.7])
    assert env.data.body("ball").xpos == pytest.approx([2.0, 1.0, BALL.radius])
    assert env.robot_pose[2] == pytest.approx(0.3)


def rollout(seed: int, steps: int = 120) -> tuple[np.ndarray, np.ndarray]:
    environment = GoToBallEnv()
    environment.action_space.seed(seed)
    obs, _ = environment.reset(seed=seed)
    observations, rewards = [obs], []
    for _ in range(steps):
        obs, reward, terminated, truncated, _ = environment.step(environment.action_space.sample())
        observations.append(obs)
        rewards.append(reward)
        if terminated or truncated:
            obs, _ = environment.reset()
    return np.array(observations), np.array(rewards)


def test_same_seed_gives_identical_episodes():
    obs_a, rewards_a = rollout(seed=11)
    obs_b, rewards_b = rollout(seed=11)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(rewards_a, rewards_b)


def test_different_seeds_give_different_starts(env):
    first, _ = env.reset(seed=1)
    second, _ = env.reset(seed=2)
    assert not np.allclose(first, second)


def test_starts_are_inside_the_field_and_apart(env):
    limit_x = FIELD.half_length - env.config.spawn_margin
    limit_y = FIELD.half_width - env.config.spawn_margin
    for seed in range(200):
        env.reset(seed=seed)
        robot, ball = env.robot_pose, env.ball_position
        for x, y in (robot[:2], ball[:2]):
            assert abs(x) <= limit_x and abs(y) <= limit_y
        assert math.dist(robot[:2], ball[:2]) >= env.config.min_spawn_distance


def test_success_when_the_front_is_at_the_ball(env):
    env.reset(seed=0, options={"robot_pose": (0, 0, 0), "ball_position": (CONTACT + 0.02, 0)})
    _, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert terminated and not truncated
    assert info["is_success"]
    assert reward > env.config.success_bonus - 0.1


def test_success_allows_a_small_heading_error(env):
    angle = math.radians(10)
    ball = (CONTACT * math.cos(angle), CONTACT * math.sin(angle))
    env.reset(seed=0, options={"robot_pose": (0, 0, 0), "ball_position": ball})
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert terminated and info["is_success"]


@pytest.mark.parametrize("bearing_deg", [25, -40, 180])
def test_no_success_when_the_ball_is_off_the_heading(env, bearing_deg):
    angle = math.radians(bearing_deg)
    # In front of the flat face for small angles, clear of the round side otherwise.
    distance = CONTACT + 0.01 if abs(bearing_deg) < 30 else ROBOT.radius + BALL.radius + 0.01
    ball = (distance * math.cos(angle), distance * math.sin(angle))
    env.reset(seed=0, options={"robot_pose": (0, 0, 0), "ball_position": ball})
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert not info["is_success"]
    assert not terminated


def test_no_success_when_the_ball_is_too_far(env):
    env.reset(seed=0, options={"robot_pose": (0, 0, 0), "ball_position": (CONTACT + 0.08, 0)})
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert not info["is_success"] and not terminated


def test_reward_is_positive_when_approaching_and_negative_when_retreating(env):
    start = {"robot_pose": (0, 0, 0), "ball_position": (3.0, 0)}
    env.reset(seed=0, options=start)
    approach = [env.step(np.array([1.0, 0.0, 0.0]))[1] for _ in range(30)]
    env.reset(seed=0, options=start)
    retreat = [env.step(np.array([-1.0, 0.0, 0.0]))[1] for _ in range(30)]
    # Progress beats the time and action penalties once the robot passes about 0.7 m/s,
    # which takes it about 0.25 s from rest.
    assert all(r > 0 for r in approach[15:])
    assert all(r < 0 for r in retreat)


def test_turning_towards_the_ball_earns_more_than_turning_away(env):
    start = {"robot_pose": (0, 0, 0), "ball_position": (0, 2.0)}
    env.reset(seed=0, options=start)
    towards = sum(env.step(np.array([0.0, 0.0, 1.0]))[1] for _ in range(15))
    env.reset(seed=0, options=start)
    away = sum(env.step(np.array([0.0, 0.0, -1.0]))[1] for _ in range(15))
    env.reset(seed=0, options=start)
    still = sum(env.step(np.zeros(3))[1] for _ in range(15))
    assert towards > still > away


def test_leaving_the_field_ends_the_episode_with_a_penalty(env):
    env.reset(seed=0, options={"robot_pose": (0, 2.8, math.pi / 2), "ball_position": (-3, -2)})
    for _ in range(50):
        _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0]))
        if terminated:
            break
    assert terminated and not truncated
    assert info["out_of_field"] and not info["is_success"]
    assert reward < -env.config.out_of_field_penalty + 1.0
    assert env.robot_pose[1] > FIELD.half_width


def test_episode_is_truncated_after_ten_seconds(env):
    env.reset(seed=0, options={"robot_pose": (-3, 0, 0), "ball_position": (3, 0)})
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        steps += 1
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert steps == 500
    assert info["time"] == pytest.approx(10.0)


def test_observation_matches_the_geometry(env):
    obs, _ = env.reset(seed=0, options={"robot_pose": (1.0, 1.0, math.pi / 2),
                                        "ball_position": (1.0, 2.0)})
    # The ball is 1 m straight ahead of a robot facing +y.
    assert obs[0] * 4.5 == pytest.approx(1.0, abs=1e-6)
    assert obs[1] * 4.5 == pytest.approx(0.0, abs=1e-6)
    assert obs[3] == pytest.approx(0.0, abs=1e-6) and obs[4] == pytest.approx(1.0)
    assert obs[10] == pytest.approx(1.0) and obs[11] == pytest.approx(0.0, abs=1e-6)
    assert env.observation_space.contains(obs)


def test_rejects_unknown_render_mode():
    with pytest.raises(ValueError):
        GoToBallEnv(render_mode="human")
