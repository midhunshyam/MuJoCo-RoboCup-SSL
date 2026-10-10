"""The compiled scene matches the Division B rule values."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_ssl import FIELD, ROBOT
from mujoco_ssl.scene import build_scene_xml, load_model

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return load_model()


def box(model: mujoco.MjModel, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Centre and half-sizes of an axis-aligned box fixed to the world."""
    geom = model.geom(name)
    assert model.geom_bodyid[geom.id] == 0
    return np.array(geom.pos), np.array(geom.size)


def test_field_lines_bound_a_9_by_6_metre_playing_area(model):
    centre, size = box(model, "touch_line_top")
    assert centre[1] == pytest.approx(3.0)
    assert 2 * size[1] == pytest.approx(0.01)
    centre, _ = box(model, "touch_line_bottom")
    assert centre[1] == pytest.approx(-3.0)
    for name, x in (("goal_line_left", -4.5), ("goal_line_right", 4.5)):
        centre, size = box(model, name)
        assert centre[0] == pytest.approx(x)
        assert 2 * size[0] == pytest.approx(0.01)
    # Lines are visual only: nothing collides with them.
    assert model.geom("touch_line_top").contype == 0
    assert model.geom("touch_line_top").conaffinity == 0


def test_walls_stand_0_3_metres_outside_the_field_lines(model):
    centre, size = box(model, "wall_right")
    assert centre[0] - size[0] == pytest.approx(4.5 + 0.3)
    centre, size = box(model, "wall_left")
    assert centre[0] + size[0] == pytest.approx(-4.5 - 0.3)
    centre, size = box(model, "wall_top")
    assert centre[1] - size[1] == pytest.approx(3.0 + 0.3)
    centre, size = box(model, "wall_bottom")
    assert centre[1] + size[1] == pytest.approx(-3.0 - 0.3)


@pytest.mark.parametrize("side", ["left", "right"])
def test_goal_dimensions(model, side):
    top_c, top_s = box(model, f"goal_{side}_side_top")
    bottom_c, bottom_s = box(model, f"goal_{side}_side_bottom")
    back_c, back_s = box(model, f"goal_{side}_back")
    inner_width = (top_c[1] - top_s[1]) - (bottom_c[1] + bottom_s[1])
    assert inner_width == pytest.approx(1.0)
    goal_line_edge = 4.5 + FIELD.line_width / 2
    inner_depth = abs(back_c[0]) - back_s[0] - goal_line_edge
    assert inner_depth == pytest.approx(0.18)
    for size in (top_s, bottom_s, back_s):
        assert 2 * size[2] == pytest.approx(0.16)
    assert 2 * top_s[1] == pytest.approx(0.02)
    assert 2 * back_s[0] == pytest.approx(0.02)
    # The side walls start at the goal line without overlapping it.
    assert abs(top_c[0]) - top_s[0] == pytest.approx(goal_line_edge)


def test_centre_circle_has_a_0_5_metre_radius(model):
    names = [model.geom(i).name for i in range(model.ngeom)]
    segments = [n for n in names if n.startswith("centre_circle_")]
    assert len(segments) >= 32
    radii = [math.hypot(*model.geom(n).pos[:2]) for n in segments]
    assert np.allclose(radii, 0.5)


@pytest.mark.parametrize("side", [-1, 1])
def test_defense_area_is_2_metres_wide_and_1_metre_deep(model, side):
    label = "left" if side < 0 else "right"
    front, front_size = box(model, f"defense_{label}_front")
    assert front[0] == pytest.approx(side * 3.5)
    assert 2 * front_size[1] == pytest.approx(2.0 + FIELD.line_width)
    for edge, y in (("top", 1.0), ("bottom", -1.0)):
        centre, size = box(model, f"defense_{label}_{edge}")
        assert centre[1] == pytest.approx(y)
        assert sorted([abs(centre[0]) - size[0], abs(centre[0]) + size[0]]) == pytest.approx(
            [3.5, 4.5]
        )


def test_ball_is_a_golf_ball(model):
    ball = model.geom("ball")
    assert ball.type == mujoco.mjtGeom.mjGEOM_SPHERE
    assert 2 * ball.size[0] == pytest.approx(0.043)
    assert model.body("ball").mass[0] == pytest.approx(0.046)
    assert ball.condim == 6  # sliding, torsional and rolling friction


def test_robot_fits_the_rule_cylinder_and_has_a_flat_front(model):
    geom = model.geom("robot")
    mesh = geom.dataid[0]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, geom.quat)
    vertices = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + geom.pos
    assert np.hypot(vertices[:, 0], vertices[:, 1]).max() <= 0.09 + 1e-6
    assert np.ptp(vertices[:, 2]) == pytest.approx(0.15)
    assert vertices[:, 0].max() == pytest.approx(ROBOT.front_distance)
    assert model.body("robot").mass[0] == pytest.approx(2.5)


def test_robot_does_not_collide_with_the_floor_but_does_with_ball_and_walls(model):
    def collides(a: str, b: str) -> bool:
        ga, gb = model.geom(a), model.geom(b)
        return bool((ga.contype & gb.conaffinity) or (gb.contype & ga.conaffinity))

    assert not collides("robot", "floor")
    assert collides("robot", "ball")
    assert collides("robot", "wall_top")
    assert collides("robot", "goal_left_back")
    assert collides("ball", "floor")
    assert collides("ball", "wall_left")


def test_exported_mjcf_matches_the_generator():
    exported = (REPO / "assets" / "ssl_field.xml").read_text()
    assert exported == build_scene_xml(), "run: python scripts/export_mjcf.py"
