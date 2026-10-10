"""Scripted go-to-ball controller, the reference point for the learned policy."""

from __future__ import annotations

import math

import numpy as np

from .field import BALL, ROBOT
from .go_to_ball import POSITION_SCALE


class GoToBallController:
    """Proportional controller on the ball position, worked out in the robot frame.

    It drives towards the point where its front face would touch the ball and turns to face
    the ball at the same time. The target stands back from the ball in proportion to the
    heading error, so the robot does not knock the ball with its side or back while turning.

    The controller reads the same observation as the policy and returns an action in
    [-1, 1]^3, so the two can be swapped in any rollout loop.
    """

    def __init__(
        self,
        position_gain: float = 5.0,
        heading_gain: float = 6.0,
        standoff_per_rad: float = 0.12,
        overreach: float = 0.04,
    ) -> None:
        self.position_gain = position_gain  # commanded speed per metre of position error, 1/s
        self.heading_gain = heading_gain  # commanded yaw rate per radian of heading error, 1/s
        self.standoff_per_rad = standoff_per_rad  # extra distance kept per radian of error, m
        self.overreach = overreach  # aim this far into the ball so the approach does not stall

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        ball_x = float(observation[0]) * POSITION_SCALE
        ball_y = float(observation[1]) * POSITION_SCALE
        distance = math.hypot(ball_x, ball_y)
        bearing = math.atan2(ball_y, ball_x)

        # Where the robot centre should be: on the line to the ball, one reach short of it.
        reach = ROBOT.front_distance + BALL.radius - self.overreach
        reach += self.standoff_per_rad * abs(bearing)
        scale = (distance - reach) / distance if distance > 1e-9 else 0.0
        vx = self.position_gain * ball_x * scale
        vy = self.position_gain * ball_y * scale
        speed = math.hypot(vx, vy)
        if speed > ROBOT.max_speed:
            vx, vy = vx * ROBOT.max_speed / speed, vy * ROBOT.max_speed / speed

        omega = float(np.clip(self.heading_gain * bearing, -ROBOT.max_angular_speed,
                              ROBOT.max_angular_speed))
        return np.array(
            [vx / ROBOT.max_speed, vy / ROBOT.max_speed, omega / ROBOT.max_angular_speed],
            dtype=np.float32,
        )
