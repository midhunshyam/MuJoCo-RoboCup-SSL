"""RoboCup Small Size League (Division B) field, ball and robot in MuJoCo, as Gymnasium tasks."""

from gymnasium.envs.registration import register, registry

from .field import BALL, FIELD, ROBOT, BallSpec, FieldSpec, RobotSpec
from .go_to_ball import GoToBallConfig, GoToBallEnv

__version__ = "0.1.0"

GO_TO_BALL_ID = "RoboCupSSL/GoToBall-v0"

if GO_TO_BALL_ID not in registry:
    register(id=GO_TO_BALL_ID, entry_point="mujoco_ssl.go_to_ball:GoToBallEnv")

__all__ = [
    "BALL",
    "FIELD",
    "GO_TO_BALL_ID",
    "ROBOT",
    "BallSpec",
    "FieldSpec",
    "GoToBallConfig",
    "GoToBallEnv",
    "RobotSpec",
]
