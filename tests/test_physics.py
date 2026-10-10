"""Ball and robot dynamics."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from mujoco_ssl import BALL, GoToBallEnv

FAR_CORNER = (4.0, 2.7)  # somewhere out of the way


@pytest.fixture
def env() -> GoToBallEnv:
    environment = GoToBallEnv()
    yield environment
    environment.close()


def ball_speed(env: GoToBallEnv) -> float:
    v = env.model.joint("ball").dofadr[0]
    return math.hypot(env.data.qvel[v], env.data.qvel[v + 1])


def launch_ball(env: GoToBallEnv, speed: float) -> None:
    env.reset(seed=0, options={"robot_pose": (0.0, 2.5, 0.0), "ball_position": (-3.0, 0.0)})
    env.data.qvel[env.model.joint("ball").dofadr[0]] = speed


def test_ball_rolls_and_comes_to_rest(env):
    launch_ball(env, 2.0)
    for _ in range(int(8.0 / env.model.opt.timestep)):
        mujoco.mj_step(env.model, env.data)
    x, y, z = env.ball_position
    travelled = x - (-3.0)
    assert 2.0 < travelled < 5.0
    assert abs(y) < 1e-3
    assert z == pytest.approx(BALL.radius, abs=1e-3)  # still on the carpet
    assert ball_speed(env) < 1e-3


def test_rolling_ball_decelerates_gently(env):
    launch_ball(env, 2.0)
    samples = []
    for step in range(int(4.0 / env.model.opt.timestep)):
        mujoco.mj_step(env.model, env.data)
        samples.append((step * env.model.opt.timestep, ball_speed(env)))
    t, v = np.array(samples).T
    rolling = (v < 1.2) & (v > 0.4)  # after the short sliding phase
    slope = np.polyfit(t[rolling], v[rolling], 1)[0]
    assert -0.6 < slope < -0.2  # m/s^2


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -2.3])
@pytest.mark.parametrize(
    ("action", "direction"),
    [((1.0, 0.0, 0.0), 0.0), ((0.0, 1.0, 0.0), math.pi / 2), ((-1.0, 0.0, 0.0), math.pi)],
)
def test_actions_move_the_robot_in_the_robot_frame(env, yaw, action, direction):
    env.reset(seed=0, options={"robot_pose": (0.0, 0.0, yaw), "ball_position": FAR_CORNER})
    for _ in range(25):  # 0.5 s
        env.step(np.array(action))
    moved = env.robot_pose[:2]
    heading = math.atan2(moved[1], moved[0])
    expected = yaw + direction
    error = math.atan2(math.sin(heading - expected), math.cos(heading - expected))
    assert np.linalg.norm(moved) > 0.3
    assert abs(error) < math.radians(2)


def test_turn_action_changes_yaw_only(env):
    env.reset(seed=0, options={"robot_pose": (0.0, 0.0, 0.0), "ball_position": FAR_CORNER})
    for _ in range(10):  # 0.2 s at up to 20 rad/s^2
        env.step(np.array([0.0, 0.0, 1.0]))
    x, y, yaw = env.robot_pose
    assert yaw > 0.3
    assert math.hypot(x, y) < 1e-3
    for _ in range(25):  # the spin takes 0.2 s to stop before it reverses
        env.step(np.array([0.0, 0.0, -1.0]))
    assert env.data.qvel[2] < -3.5


def test_speed_and_acceleration_limits(env):
    env.reset(seed=0, options={"robot_pose": (-3.0, 0.0, 0.0), "ball_position": FAR_CORNER})
    speeds = []
    for _ in range(75):
        env.step(np.array([1.0, 1.0, 0.0]))  # diagonal: clipped to 2 m/s in norm
        speeds.append(math.hypot(env.data.qvel[0], env.data.qvel[1]))
    assert speeds[9] < 0.2 * 3.0 * math.sqrt(2) + 0.05  # 0.2 s at no more than 3 m/s^2 per axis
    assert max(speeds) <= 2.0 + 1e-3
    assert speeds[-1] == pytest.approx(2.0, abs=0.01)
    env.reset(seed=0, options={"robot_pose": (0.0, 0.0, 0.0), "ball_position": FAR_CORNER})
    for _ in range(50):
        env.step(np.array([0.0, 0.0, 1.0]))
    assert abs(env.data.qvel[2]) == pytest.approx(4.0, abs=0.01)


def test_robot_pushes_the_ball(env):
    env.reset(seed=0, options={"robot_pose": (0.0, 0.0, 0.0), "ball_position": (0.0, 0.25)})
    for _ in range(25):
        env.step(np.array([0.0, 1.0, 0.0]))  # drive sideways into the ball
    assert env.ball_position[1] > 0.4
