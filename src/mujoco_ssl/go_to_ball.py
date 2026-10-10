"""GoToBall: drive an SSL robot to the ball and face it.

The robot and the ball start at random places on the field. The episode ends with success
when the centre of the robot's flat front is within 3 cm of the ball's surface and the
ball lies within 15 degrees of the robot's heading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .field import BALL, FIELD, ROBOT
from .scene import load_model

# Observation scales: positions by half the field length, velocities by the robot's limits.
POSITION_SCALE = FIELD.half_length
SPEED_SCALE = ROBOT.max_speed
TURN_SCALE = ROBOT.max_angular_speed

OBSERVATION_NAMES = (
    "ball_x",  # ball centre relative to the robot centre, robot frame
    "ball_y",
    "ball_distance",  # robot centre to ball centre
    "ball_bearing_sin",  # direction of the ball in the robot frame (0 = straight ahead)
    "ball_bearing_cos",
    "ball_vx",  # ball velocity, robot frame
    "ball_vy",
    "robot_vx",  # robot velocity, robot frame
    "robot_vy",
    "robot_yaw_rate",
    "robot_heading_sin",  # robot heading in the field frame
    "robot_heading_cos",
    "robot_x",  # robot position on the field
    "robot_y",
)


# Generous bounds; observations are clipped to them, which only matters if the ball is
# knocked very hard.
OBSERVATION_HIGH = np.array([5, 5, 5, 1, 1, 10, 10, 5, 5, 5, 1, 1, 2, 2], dtype=np.float32)
OBSERVATION_LOW = -OBSERVATION_HIGH
OBSERVATION_LOW[2] = 0.0  # a distance


@dataclass(frozen=True)
class GoToBallConfig:
    """Task settings. Distances in metres, times in seconds."""

    control_frequency: float = 50.0  # Hz
    episode_seconds: float = 10.0
    success_gap: float = 0.03  # centre of the front face to the ball's surface
    success_heading_deg: float = 15.0
    spawn_margin: float = 0.3  # robot and ball start at least this far inside the field lines
    min_spawn_distance: float = 0.5  # robot centre to ball centre at the start
    progress_weight: float = 1.0  # reward per metre of gap closed
    heading_weight: float = 0.5  # reward per pi rad of heading error removed
    success_bonus: float = 10.0
    out_of_field_penalty: float = 5.0
    time_penalty: float = 0.01  # per control step
    action_penalty: float = 0.01  # times the mean squared action


class Snapshot(NamedTuple):
    """Robot and ball state, with the ball described in the robot frame."""

    robot_x: float
    robot_y: float
    yaw: float
    ball_x: float  # ball relative to the robot, robot frame
    ball_y: float
    distance: float  # robot centre to ball centre
    bearing: float  # angle of the ball from the robot's heading, radians
    gap: float  # centre of the front face to the ball's surface
    ball_vx: float  # ball velocity, robot frame
    ball_vy: float
    robot_vx: float  # robot velocity, robot frame
    robot_vy: float
    yaw_rate: float
    ball_world_x: float
    ball_world_y: float


class GoToBallEnv(gym.Env):
    """Drive the robot to the ball and face it.

    Actions are in [-1, 1]^3 and scale to a velocity command in the robot frame: forward
    speed, leftward speed (together clipped to 2 m/s) and yaw rate (up to 4 rad/s).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        config: GoToBallConfig | None = None,
        render_width: int = 720,
        render_height: int = 512,
        camera: str = "top",
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.config = config or GoToBallConfig()
        self.model = load_model()
        self.data = mujoco.MjData(self.model)

        self.frame_skip = round(1.0 / (self.config.control_frequency * self.model.opt.timestep))
        self.dt = self.frame_skip * self.model.opt.timestep
        self.max_steps = round(self.config.episode_seconds / self.dt)
        self.metadata = {**self.metadata, "render_fps": round(1.0 / self.dt)}

        robot_joints = [self.model.joint(n).qposadr[0] for n in ("robot_x", "robot_y", "robot_yaw")]
        if robot_joints != [0, 1, 2]:
            raise RuntimeError("expected the robot's x, y and yaw joints first in qpos")
        self._ball_qpos = self.model.joint("ball").qposadr[0]
        self._ball_dof = self.model.joint("ball").dofadr[0]
        self._robot_geom = self.model.geom("robot").id
        floor = self.model.geom("floor").id
        self._wall_geoms = np.array([
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == 0 and self.model.geom_contype[g] and g != floor
        ])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(OBSERVATION_LOW, OBSERVATION_HIGH, dtype=np.float32)

        self.render_width, self.render_height, self.camera = render_width, render_height, camera
        self._renderer: mujoco.Renderer | None = None
        self._steps = 0
        self._potential = 0.0
        self._ball_start = np.zeros(2)

    # ---------------------------------------------------------------- gymnasium API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode.

        `options` may fix the start: {"robot_pose": (x, y, yaw), "ball_position": (x, y)}.
        Anything not given is drawn at random.
        """
        super().reset(seed=seed)
        options = options or {}
        robot, ball = self._sample_start(options.get("robot_pose"), options.get("ball_position"))

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = robot
        b = self._ball_qpos
        self.data.qpos[b:b + 7] = (ball[0], ball[1], BALL.radius, 1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._ball_start = np.array(ball, dtype=float)
        snap = self._snapshot()
        self._potential = self._potential_of(snap)
        return self._observation(snap), self._info(snap, success=False, out=False, wall=False)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        vx, vy = a[0] * ROBOT.max_speed, a[1] * ROBOT.max_speed
        speed = math.hypot(vx, vy)
        if speed > ROBOT.max_speed:
            vx, vy = vx * ROBOT.max_speed / speed, vy * ROBOT.max_speed / speed
        self.data.ctrl[:] = (vx, vy, a[2] * ROBOT.max_angular_speed)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._steps += 1

        snap = self._snapshot()
        success = self._is_success(snap)
        out = not success and self._is_out_of_field(snap)
        wall = self._robot_touches_wall()

        potential = self._potential_of(snap)
        reward = potential - self._potential
        reward -= cfg.time_penalty + cfg.action_penalty * float(np.mean(a**2))
        if success:
            reward += cfg.success_bonus
        if out:
            reward -= cfg.out_of_field_penalty
        self._potential = potential

        terminated = success or out
        truncated = not terminated and self._steps >= self.max_steps
        info = self._info(snap, success=success, out=out, wall=wall)
        return self._observation(snap), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---------------------------------------------------------------- helpers

    @property
    def robot_pose(self) -> np.ndarray:
        """Robot x, y (metres) and yaw (radians) in the field frame."""
        return self.data.qpos[0:3].copy()

    @property
    def ball_position(self) -> np.ndarray:
        """Ball centre x, y, z in the field frame."""
        b = self._ball_qpos
        return self.data.qpos[b:b + 3].copy()

    def _sample_start(self, robot_pose, ball_position) -> tuple[np.ndarray, np.ndarray]:
        cfg, rng = self.config, self.np_random
        half_x = FIELD.half_length - cfg.spawn_margin
        half_y = FIELD.half_width - cfg.spawn_margin
        if robot_pose is None:
            robot = np.array([rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y),
                              rng.uniform(-math.pi, math.pi)])
        else:
            robot = np.asarray(robot_pose, dtype=float).reshape(3)
        if ball_position is not None:
            return robot, np.asarray(ball_position, dtype=float).reshape(2)
        while True:
            ball = np.array([rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y)])
            if math.hypot(ball[0] - robot[0], ball[1] - robot[1]) >= cfg.min_spawn_distance:
                return robot, ball

    def _snapshot(self) -> Snapshot:
        qpos, qvel = self.data.qpos, self.data.qvel
        x, y, yaw = float(qpos[0]), float(qpos[1]), float(qpos[2])
        c, s = math.cos(yaw), math.sin(yaw)
        b, v = self._ball_qpos, self._ball_dof
        ball_world_x, ball_world_y = float(qpos[b]), float(qpos[b + 1])
        dx, dy = ball_world_x - x, ball_world_y - y
        ball_x, ball_y = c * dx + s * dy, -s * dx + c * dy
        ball_vx, ball_vy = qvel[v], qvel[v + 1]  # free joint: linear velocity in the world frame
        robot_vx, robot_vy = qvel[0], qvel[1]
        return Snapshot(
            robot_x=x,
            robot_y=y,
            yaw=yaw,
            ball_x=ball_x,
            ball_y=ball_y,
            distance=math.hypot(ball_x, ball_y),
            bearing=math.atan2(ball_y, ball_x),
            gap=math.hypot(ball_x - ROBOT.front_distance, ball_y) - BALL.radius,
            ball_vx=c * ball_vx + s * ball_vy,
            ball_vy=-s * ball_vx + c * ball_vy,
            robot_vx=c * robot_vx + s * robot_vy,
            robot_vy=-s * robot_vx + c * robot_vy,
            yaw_rate=float(qvel[2]),
            ball_world_x=ball_world_x,
            ball_world_y=ball_world_y,
        )

    def _observation(self, snap: Snapshot) -> np.ndarray:
        obs = np.array(
            [
                snap.ball_x / POSITION_SCALE,
                snap.ball_y / POSITION_SCALE,
                snap.distance / POSITION_SCALE,
                math.sin(snap.bearing),
                math.cos(snap.bearing),
                snap.ball_vx / SPEED_SCALE,
                snap.ball_vy / SPEED_SCALE,
                snap.robot_vx / SPEED_SCALE,
                snap.robot_vy / SPEED_SCALE,
                snap.yaw_rate / TURN_SCALE,
                math.sin(snap.yaw),
                math.cos(snap.yaw),
                snap.robot_x / FIELD.half_length,
                snap.robot_y / FIELD.half_width,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, OBSERVATION_LOW, OBSERVATION_HIGH)

    def _potential_of(self, snap: Snapshot) -> float:
        cfg = self.config
        return -(cfg.progress_weight * snap.gap + cfg.heading_weight * abs(snap.bearing) / math.pi)

    def _is_success(self, snap: Snapshot) -> bool:
        cfg = self.config
        return snap.gap <= cfg.success_gap and abs(snap.bearing) <= math.radians(
            cfg.success_heading_deg
        )

    def _is_out_of_field(self, snap: Snapshot) -> bool:
        """Robot centre past a field line, or the ball fully over one."""
        hx, hy = FIELD.half_length, FIELD.half_width
        if abs(snap.robot_x) > hx or abs(snap.robot_y) > hy:
            return True
        margin = FIELD.line_width / 2 + BALL.radius
        return abs(snap.ball_world_x) > hx + margin or abs(snap.ball_world_y) > hy + margin

    def _robot_touches_wall(self) -> bool:
        n = self.data.ncon
        if n == 0:
            return False
        pairs = np.stack([self.data.contact.geom1[:n], self.data.contact.geom2[:n]], axis=1)
        has_robot = (pairs == self._robot_geom).any(axis=1)
        has_wall = np.isin(pairs, self._wall_geoms).any(axis=1)
        return bool(np.any(has_robot & has_wall))

    def _info(self, snap: Snapshot, *, success: bool, out: bool, wall: bool) -> dict[str, Any]:
        moved = math.hypot(snap.ball_world_x - self._ball_start[0],
                           snap.ball_world_y - self._ball_start[1])
        return {
            "is_success": success,
            "out_of_field": out,
            "wall_contact": wall,
            "gap": snap.gap,
            "heading_error_deg": math.degrees(snap.bearing),
            "time": self._steps * self.dt,
            "ball_displacement": moved,
        }
