"""Measure how the ball and the robot move in the model.

    python scripts/measure_dynamics.py

Writes results/dynamics.json, the source of the measured values in the README.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from mujoco_ssl import BALL, GoToBallEnv

OUT_OF_THE_WAY = {"robot_pose": (0.0, 2.5, 0.0), "ball_position": (-3.0, 0.0)}


def ball_roll(env: GoToBallEnv, launch_speed: float = 2.0) -> dict:
    """Launch the ball along +x without spin and follow it until it stops."""
    env.reset(seed=0, options=OUT_OF_THE_WAY)
    joint = env.model.joint("ball")
    q, v = joint.qposadr[0], joint.dofadr[0]
    env.data.qvel[v] = launch_speed
    dt = env.model.opt.timestep
    times, speeds, spins, slips, heights = [], [], [], [], []
    spin = np.zeros(3)
    for i in range(int(12.0 / dt)):
        mujoco.mj_step(env.model, env.data)
        mujoco.mju_rotVecQuat(spin, env.data.qvel[v + 3:v + 6], env.data.qpos[q + 3:q + 7])
        vx, vy = env.data.qvel[v], env.data.qvel[v + 1]
        # Velocity of the ball's surface at the contact point: v - r * (w x z).
        slips.append(math.hypot(vx - BALL.radius * spin[1], vy + BALL.radius * spin[0]))
        times.append((i + 1) * dt)
        speeds.append(math.hypot(vx, vy))
        spins.append(float(np.linalg.norm(spin)))
        heights.append(env.data.qpos[q + 2])
    t, speed, spin_rate, slip, z = map(np.array, (times, speeds, spins, slips, heights))

    # Sliding friction spins the ball up; once the spin peaks, rolling resistance takes over.
    rolling_from = int(np.argmax(spin_rate))
    rolling = (t > t[rolling_from] + 0.1) & (speed > 0.2)
    rolling_deceleration = -np.polyfit(t[rolling], speed[rolling], 1)[0]
    stopped = int(np.argmax(speed < 1e-3))
    return {
        "launch_speed_m_s": launch_speed,
        "spin_up_time_s": round(float(t[rolling_from]), 3),
        "speed_after_spin_up_m_s": round(float(speed[rolling_from]), 3),
        "rolling_deceleration_m_s2": round(float(rolling_deceleration), 3),
        "slip_while_rolling_fraction_of_speed": round(
            float(np.median(slip[rolling] / speed[rolling])), 3
        ),
        "distance_to_rest_m": round(float(env.ball_position[0] - (-3.0)), 3),
        "time_to_rest_s": round(float(t[stopped]), 2),
        "max_rise_while_spinning_up_mm": round(
            float((z[:rolling_from].max() - BALL.radius) * 1e3), 2
        ),
        "max_rise_while_rolling_mm": round(float((z[rolling].max() - BALL.radius) * 1e3), 2),
    }


def robot_response(env: GoToBallEnv) -> dict:
    """Full forward command from rest, then a full turn command from rest."""
    env.reset(seed=0, options={"robot_pose": (-3.0, 0.0, 0.0), "ball_position": (3.0, 2.0)})
    speeds, positions = [], []
    for _ in range(100):
        env.step(np.array([1.0, 0.0, 0.0]))
        speeds.append(math.hypot(env.data.qvel[0], env.data.qvel[1]))
        positions.append(env.robot_pose[0])
    speeds = np.array(speeds)
    t = env.dt * np.arange(1, len(speeds) + 1)
    one_second = round(1.0 / env.dt) - 1

    # Diagonal in the robot frame: both axis limits are active at once.
    env.reset(seed=0, options={"robot_pose": (-3.0, -2.0, 0.0), "ball_position": (3.0, 2.0)})
    for _ in range(10):
        env.step(np.array([1.0, 1.0, 0.0]))
    diagonal_speed = math.hypot(env.data.qvel[0], env.data.qvel[1])

    env.reset(seed=0, options={"robot_pose": (0.0, 0.0, 0.0), "ball_position": (3.0, 2.0)})
    rates = []
    for _ in range(50):
        env.step(np.array([0.0, 0.0, 1.0]))
        rates.append(env.data.qvel[2])
    rates = np.array(rates)
    return {
        "top_speed_m_s": round(float(speeds.max()), 3),
        "mean_acceleration_first_0.2_s_m_s2": round(float(speeds[9] / 0.2), 2),
        "mean_acceleration_first_0.2_s_diagonal_m_s2": round(float(diagonal_speed / 0.2), 2),
        "time_to_95_percent_of_top_speed_s": round(float(t[np.argmax(speeds > 0.95 * 2.0)]), 2),
        "distance_in_first_second_m": round(float(positions[one_second] - (-3.0)), 3),
        "top_yaw_rate_rad_s": round(float(rates.max()), 3),
        "time_to_95_percent_of_top_yaw_rate_s": round(
            float(env.dt * (np.argmax(rates > 0.95 * 4.0) + 1)), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/dynamics.json"))
    args = parser.parse_args()
    env = GoToBallEnv()
    result = {"ball": ball_roll(env), "robot": robot_response(env)}
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
