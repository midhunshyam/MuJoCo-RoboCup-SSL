"""Build the MJCF scene: a Division B field, one robot and the ball.

`build_scene_xml()` returns the model as an XML string and `load_model()` compiles it.
The copy in `assets/ssl_field.xml` is written by `scripts/export_mjcf.py`.

Frame: the origin is the centre of the field, x runs from the left goal to the right goal,
y runs across the field and z points up.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco

from .field import BALL, FIELD, ROBOT, BallSpec, FieldSpec, RobotSpec

TIMESTEP = 0.002  # physics step in seconds (500 Hz)

# Collision filtering. MuJoCo lets two geoms touch when (contype1 & conaffinity2) or
# (contype2 & conaffinity1) is non-zero. The robot rides on planar joints, so it ignores
# the floor, but it still hits the ball, the walls and (later) other robots.
FLOOR_BITS = {"contype": "1", "conaffinity": "1"}
SOLID_BITS = {"contype": "1", "conaffinity": "3"}  # ball, boundary walls, goal walls
ROBOT_BITS = {"contype": "2", "conaffinity": "2"}
VISUAL_ONLY = {"contype": "0", "conaffinity": "0"}

LINE_LIFT = 0.0005  # field lines sit half a millimetre above the carpet so they render on top
CIRCLE_SEGMENTS = 64
ROBOT_SEGMENTS = 48

MATERIALS = {
    "carpet": {"rgba": "0.17 0.47 0.23 1", "specular": "0", "shininess": "0"},
    "surround": {"rgba": "0.16 0.17 0.18 1", "specular": "0"},
    "line": {"rgba": "0.96 0.96 0.96 1", "emission": "0.25", "specular": "0"},
    "wall": {"rgba": "0.07 0.07 0.08 1"},
    "goal_inner": {"rgba": "0.93 0.93 0.93 1"},
    "ball": {"rgba": "1 0.45 0.05 1", "specular": "0.4", "shininess": "0.6", "emission": "0.15"},
    "robot_shell": {"rgba": "0.09 0.09 0.1 1", "specular": "0.25"},
    "robot_front": {"rgba": "0.78 0.79 0.8 1"},
    "team_blue": {"rgba": "0.12 0.4 1 1", "emission": "0.2"},
    "marker_pink": {"rgba": "1 0.3 0.7 1", "emission": "0.2"},
    "marker_green": {"rgba": "0.25 0.9 0.35 1", "emission": "0.2"},
}


def _num(*values: float) -> str:
    """Format numbers for the XML: six significant digits and no negative zero."""
    return " ".join(f"{value + 0.0:.6g}" for value in values)


def _xyaxes_looking_at(position: tuple, target: tuple) -> str:
    """Camera axes for a camera at `position` looking at `target`, with a level x axis."""
    forward = [t - p for p, t in zip(position, target)]
    norm = math.sqrt(sum(f * f for f in forward))
    back = [-f / norm for f in forward]  # MuJoCo cameras look along their -z axis
    x_axis = (-back[1], back[0], 0.0)
    x_norm = math.hypot(x_axis[0], x_axis[1])
    x_axis = (x_axis[0] / x_norm, x_axis[1] / x_norm, 0.0)
    y_axis = (
        back[1] * x_axis[2] - back[2] * x_axis[1],
        back[2] * x_axis[0] - back[0] * x_axis[2],
        back[0] * x_axis[1] - back[1] * x_axis[0],
    )
    return _num(*x_axis, *y_axis)


def _robot_vertices(robot: RobotSpec) -> str:
    """Vertices of a cylinder with its front cut flat; MuJoCo takes their convex hull."""
    cut = math.acos(robot.front_distance / robot.radius)
    angles = [cut + (2 * math.pi - 2 * cut) * i / ROBOT_SEGMENTS for i in range(ROBOT_SEGMENTS + 1)]
    points = []
    for z in (-robot.height / 2, robot.height / 2):
        for angle in angles:
            points += [robot.radius * math.cos(angle), robot.radius * math.sin(angle), z]
    return _num(*points)


def _header(root: ET.Element, field: FieldSpec) -> None:
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=_num(TIMESTEP), integrator="implicitfast")
    # Size of the scene for rendering only (shadow map and clipping planes). Left to MuJoCo,
    # the large surround plane would spread the shadow map thinly and blur the shadows.
    extent = math.hypot(field.half_length + field.surface_margin,
                        field.half_width + field.surface_margin)
    ET.SubElement(root, "statistic", extent=_num(round(extent, 2)), center="0 0 0")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")
    ET.SubElement(visual, "headlight", ambient="0.32 0.32 0.32", diffuse="0.42 0.42 0.42",
                  specular="0 0 0")


def _assets(root: ET.Element, robot: RobotSpec) -> None:
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.3 0.33 0.37", rgb2="0.1 0.11 0.12", width="256", height="256")
    for name, attributes in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, **attributes)
    ET.SubElement(asset, "mesh", name="robot_shell", vertex=_robot_vertices(robot))


def _lights_and_cameras(world: ET.Element, field: FieldSpec) -> None:
    ET.SubElement(world, "light", name="sun", directional="true", pos="0 0 8",
                  dir="0.25 0.35 -1", diffuse="0.55 0.55 0.55", specular="0.1 0.1 0.1",
                  castshadow="true")
    # Top-down view of the whole field surface.
    height = 10.0
    half_view = field.half_width + field.surface_margin + 0.1
    fovy = math.degrees(2 * math.atan(half_view / height))
    ET.SubElement(world, "camera", name="top", pos=_num(0, 0, height), xyaxes="1 0 0 0 1 0",
                  fovy=_num(round(fovy, 2)))
    # Raised view from behind the bottom touch line, like a broadcast camera.
    position = (0.0, -field.half_width - 5.2, 6.4)
    ET.SubElement(world, "camera", name="broadcast", pos=_num(*position),
                  xyaxes=_xyaxes_looking_at(position, (0.0, -0.4, 0.0)), fovy="45")


def _floor(world: ET.Element, field: FieldSpec) -> None:
    half_x = field.half_length + field.surface_margin
    half_y = field.half_width + field.surface_margin
    ET.SubElement(world, "geom", name="floor", type="plane", size=_num(half_x, half_y, 0.1),
                  material="carpet", **FLOOR_BITS)
    ET.SubElement(world, "geom", name="surround", type="plane", pos=_num(0, 0, -0.002),
                  size=_num(3 * half_x, 3 * half_y, 0.1), material="surround", **VISUAL_ONLY)


def _line(world: ET.Element, name: str, centre: tuple, half_length: float, along_x: bool,
          width: float) -> None:
    """A straight field line, aligned with the x or the y axis."""
    size = (half_length, width / 2) if along_x else (width / 2, half_length)
    ET.SubElement(world, "geom", name=name, type="box", pos=_num(*centre, LINE_LIFT),
                  size=_num(*size, LINE_LIFT), material="line", **VISUAL_ONLY)


def _field_lines(world: ET.Element, field: FieldSpec) -> None:
    hl, hw, w = field.half_length, field.half_width, field.line_width
    # Field lines bounding the playing area, extended by half a line width to close the corners.
    _line(world, "touch_line_top", (0, hw), hl + w / 2, True, w)
    _line(world, "touch_line_bottom", (0, -hw), hl + w / 2, True, w)
    _line(world, "goal_line_left", (-hl, 0), hw + w / 2, False, w)
    _line(world, "goal_line_right", (hl, 0), hw + w / 2, False, w)
    _line(world, "halfway_line", (0, 0), hw, False, w)
    _line(world, "goal_to_goal_line", (0, 0), hl, True, w)

    # Centre circle as short straight segments, centred on the circle.
    radius = field.centre_circle_radius
    half_segment = radius * math.tan(math.pi / CIRCLE_SEGMENTS)
    for i in range(CIRCLE_SEGMENTS):
        angle = 2 * math.pi * i / CIRCLE_SEGMENTS
        ET.SubElement(world, "geom", name=f"centre_circle_{i:02d}", type="box",
                      pos=_num(radius * math.cos(angle), radius * math.sin(angle), LINE_LIFT),
                      size=_num(half_segment, w / 2, LINE_LIFT),
                      euler=_num(0, 0, angle + math.pi / 2), material="line", **VISUAL_ONLY)

    # Defence areas: rectangles touching each goal line centrally.
    depth, half_span = field.defense_area_depth, field.defense_area_width / 2
    for side, label in ((-1, "left"), (1, "right")):
        front_x, side_x = side * (hl - depth), side * (hl - depth / 2)
        _line(world, f"defense_{label}_front", (front_x, 0), half_span + w / 2, False, w)
        for y, edge in ((half_span, "top"), (-half_span, "bottom")):
            _line(world, f"defense_{label}_{edge}", (side_x, y), depth / 2, True, w)


def _box(parent: ET.Element, name: str, centre: tuple, half_size: tuple, material: str,
         bits: dict) -> None:
    ET.SubElement(parent, "geom", name=name, type="box", pos=_num(*centre),
                  size=_num(*half_size), material=material, **bits)


def _boundary_walls(world: ET.Element, field: FieldSpec) -> None:
    inner_x = field.half_length + field.margin
    inner_y = field.half_width + field.margin
    t, h = field.wall_thickness, field.wall_height
    for sign, label in ((1, "top"), (-1, "bottom")):
        _box(world, f"wall_{label}", (0, sign * (inner_y + t / 2), h / 2),
             (inner_x + t, t / 2, h / 2), "wall", SOLID_BITS)
    for sign, label in ((-1, "left"), (1, "right")):
        _box(world, f"wall_{label}", (sign * (inner_x + t / 2), 0, h / 2),
             (t / 2, inner_y, h / 2), "wall", SOLID_BITS)


def _goals(world: ET.Element, field: FieldSpec) -> None:
    """Goals stand just outside the goal line: side walls joined by a rear wall."""
    t, h, depth = field.goal_wall_thickness, field.goal_wall_height, field.goal_depth
    half_mouth = field.goal_width / 2
    start = field.half_length + field.line_width / 2  # outer edge of the goal line
    plate, plate_h = 0.0006, (h - 0.004) / 2  # white inner faces, kept below the black tops
    for side, label in ((-1, "left"), (1, "right")):
        name = f"goal_{label}"
        for y_sign, edge in ((1, "top"), (-1, "bottom")):
            _box(world, f"{name}_side_{edge}",
                 (side * (start + (depth + t) / 2), y_sign * (half_mouth + t / 2), h / 2),
                 ((depth + t) / 2, t / 2, h / 2), "wall", SOLID_BITS)
            _box(world, f"{name}_side_{edge}_inner",
                 (side * (start + depth / 2), y_sign * (half_mouth - plate / 2), plate_h),
                 (depth / 2, plate, plate_h), "goal_inner", VISUAL_ONLY)
        _box(world, f"{name}_back", (side * (start + depth + t / 2), 0, h / 2),
             (t / 2, half_mouth + t, h / 2), "wall", SOLID_BITS)
        _box(world, f"{name}_back_inner", (side * (start + depth - plate / 2), 0, plate_h),
             (plate, half_mouth, plate_h), "goal_inner", VISUAL_ONLY)


def _ball(world: ET.Element, ball: BallSpec) -> None:
    body = ET.SubElement(world, "body", name="ball", pos=_num(0.5, 0, ball.radius))
    ET.SubElement(body, "freejoint", name="ball")
    # priority="1" makes the ball's friction and condim govern all of its contacts.
    ET.SubElement(body, "geom", name="ball", type="sphere", size=_num(ball.radius),
                  mass=_num(ball.mass), material="ball", condim="6", priority="1",
                  friction=_num(ball.sliding_friction, 0.005, ball.rolling_friction),
                  **SOLID_BITS)


def _robot(world: ET.Element, robot: RobotSpec) -> None:
    half_h = robot.height / 2
    # The body sits at the origin so that the slide joints' qpos equals the field position.
    body = ET.SubElement(world, "body", name="robot", pos=_num(0, 0, half_h))
    ET.SubElement(body, "joint", name="robot_x", type="slide", axis="1 0 0")
    ET.SubElement(body, "joint", name="robot_y", type="slide", axis="0 1 0")
    ET.SubElement(body, "joint", name="robot_yaw", type="hinge", axis="0 0 1")
    side_inertia = robot.mass * (3 * robot.radius**2 + robot.height**2) / 12
    ET.SubElement(body, "inertial", pos="0 0 0", mass=_num(robot.mass),
                  diaginertia=_num(side_inertia, side_inertia, robot.yaw_inertia))
    ET.SubElement(body, "geom", name="robot", type="mesh", mesh="robot_shell",
                  material="robot_shell", friction="0.5 0.005 0.0001", **ROBOT_BITS)

    # Visual details: a light plate on the flat front, a light bar along the front edge of the
    # top, and colour markers in the style of the SSL vision pattern.
    front, half_span = robot.front_distance, robot.front_half_width
    _box(body, "robot_front_plate", (front + 0.0004, 0, 0),
         (0.0006, half_span - 0.004, half_h - 0.012), "robot_front", VISUAL_ONLY)
    _box(body, "robot_front_bar", (front - 0.007, 0, half_h + 0.0005),
         (0.006, half_span - 0.006, 0.0005), "robot_front", VISUAL_ONLY)
    markers = [
        ("robot_marker_team", (0, 0), 0.025, "team_blue"),
        ("robot_marker_front_left", (0.035, 0.0545), 0.02, "marker_pink"),
        ("robot_marker_front_right", (0.035, -0.0545), 0.02, "marker_green"),
        ("robot_marker_back_left", (-0.0545, 0.035), 0.02, "marker_green"),
        ("robot_marker_back_right", (-0.0545, -0.035), 0.02, "marker_pink"),
    ]
    for name, (x, y), radius, material in markers:
        ET.SubElement(body, "geom", name=name, type="cylinder", pos=_num(x, y, half_h + 0.0005),
                      size=_num(radius, 0.0005), material=material, **VISUAL_ONLY)
    ET.SubElement(body, "site", name="robot_centre", size="0.005", rgba="0 0 0 0")
    position = (0.0, -1.0, 0.9)
    ET.SubElement(body, "camera", name="follow", mode="trackcom", pos=_num(*position),
                  xyaxes=_xyaxes_looking_at(position, (0.0, 0.0, -half_h)), fovy="50")


def _actuators(root: ET.Element, robot: RobotSpec) -> None:
    """Velocity servos in the robot frame. The force limits bound the acceleration."""
    actuator = ET.SubElement(root, "actuator")
    linear_kv = robot.mass / robot.servo_time_constant
    linear_force = robot.mass * robot.max_acceleration
    yaw_kv = robot.yaw_inertia / robot.servo_time_constant
    yaw_torque = robot.yaw_inertia * robot.max_angular_acceleration
    servos = [
        ("forward", "1 0 0 0 0 0", linear_kv, linear_force, robot.max_speed),
        ("left", "0 1 0 0 0 0", linear_kv, linear_force, robot.max_speed),
        ("turn", "0 0 0 0 0 1", yaw_kv, yaw_torque, robot.max_angular_speed),
    ]
    for name, gear, kv, force, limit in servos:
        ET.SubElement(actuator, "velocity", name=name, site="robot_centre", gear=gear,
                      kv=_num(kv), ctrlrange=_num(-limit, limit), forcerange=_num(-force, force))


def build_scene_xml(field: FieldSpec = FIELD, ball: BallSpec = BALL,
                    robot: RobotSpec = ROBOT) -> str:
    """Return the MJCF for the field, the ball and one robot."""
    root = ET.Element("mujoco", model="ssl_division_b")
    root.append(ET.Comment(" Exported from src/mujoco_ssl/scene.py; edit that file instead. "))
    _header(root, field)
    _assets(root, robot)
    world = ET.SubElement(root, "worldbody")
    _lights_and_cameras(world, field)
    _floor(world, field)
    _field_lines(world, field)
    _boundary_walls(world, field)
    _goals(world, field)
    _robot(world, robot)  # first, so qpos[0:3] is the robot's x, y and yaw
    _ball(world, ball)
    _actuators(root, robot)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def load_model(field: FieldSpec = FIELD, ball: BallSpec = BALL,
               robot: RobotSpec = ROBOT) -> mujoco.MjModel:
    """Compile the scene into a MuJoCo model."""
    return mujoco.MjModel.from_xml_string(build_scene_xml(field, ball, robot))
