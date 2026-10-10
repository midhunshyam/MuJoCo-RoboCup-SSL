"""Dimensions of the field, the ball and the robot.

Field and ball values follow the RoboCup Small Size League rules for Division B
(https://robocup-ssl.github.io/ssl-rules/sslrules.html). Where the rules leave a value
open, the comment says "chosen here". All lengths are in metres, masses in kilograms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    """Division B field. Distances between lines are measured from line centres."""

    length: float = 9.0  # playing area, goal line to goal line
    width: float = 6.0  # playing area, touch line to touch line
    margin: float = 0.3  # from the field lines to the surrounding walls
    surface_margin: float = 0.7  # carpet continues at least this far beyond the lines
    line_width: float = 0.01
    centre_circle_radius: float = 0.5
    defense_area_width: float = 2.0  # along the goal line
    defense_area_depth: float = 1.0  # into the field
    goal_width: float = 1.0  # between the inner faces of the side walls
    goal_depth: float = 0.18
    goal_wall_height: float = 0.16
    goal_wall_thickness: float = 0.02
    wall_height: float = 0.1  # chosen here: the rules give no height for the boundary walls
    wall_thickness: float = 0.05  # chosen here

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2


@dataclass(frozen=True)
class BallSpec:
    """Standard orange golf ball."""

    radius: float = 0.0215  # 0.043 m diameter
    mass: float = 0.046
    sliding_friction: float = 0.35  # chosen here
    rolling_friction: float = 0.0012  # chosen here; MuJoCo rolling coefficient, in metres


@dataclass(frozen=True)
class RobotSpec:
    """Holonomic robot that fits the 0.18 m wide, 0.15 m high cylinder of the rules."""

    radius: float = 0.09
    height: float = 0.15
    front_distance: float = 0.075  # chosen here: centre to the flat front face
    mass: float = 2.5  # chosen here
    max_speed: float = 2.0  # chosen here, m/s
    max_angular_speed: float = 4.0  # chosen here, rad/s
    max_acceleration: float = 3.0  # chosen here, m/s^2 along each robot axis
    max_angular_acceleration: float = 20.0  # chosen here, rad/s^2
    servo_time_constant: float = 0.05  # chosen here, s; velocity servo response

    @property
    def front_half_width(self) -> float:
        """Half the width of the flat front face."""
        return math.sqrt(self.radius**2 - self.front_distance**2)

    @property
    def yaw_inertia(self) -> float:
        """Moment of inertia about the vertical axis, treating the robot as a solid cylinder."""
        return 0.5 * self.mass * self.radius**2


FIELD = FieldSpec()
BALL = BallSpec()
ROBOT = RobotSpec()
