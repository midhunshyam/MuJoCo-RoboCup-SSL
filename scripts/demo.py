"""Watch the trained policy (or the scripted baseline) in the MuJoCo viewer.

    python scripts/demo.py               # Linux and Windows
    mjpython scripts/demo.py             # macOS: the passive viewer needs mjpython
    python scripts/demo.py --baseline    # the scripted controller instead

A new random episode starts each time an episode ends. Close the window to stop.
`mjpython` is installed with the mujoco package.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from mujoco_ssl import GoToBallEnv
from mujoco_ssl.baseline import GoToBallController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/ppo_go_to_ball.zip"))
    parser.add_argument("--baseline", action="store_true", help="use the scripted controller")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    parser.add_argument("--camera", choices=["broadcast", "top", "follow"], default="broadcast")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    args = parser.parse_args()

    if args.baseline:
        policy = GoToBallController()
    else:
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")

        def policy(obs):
            return model.predict(obs, deterministic=True)[0]

    env = GoToBallEnv()
    obs, _ = env.reset(seed=args.seed)
    episode = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = env.model.camera(args.camera).id
        while viewer.is_running():
            started = time.perf_counter()
            obs, _, terminated, truncated, info = env.step(policy(obs))
            viewer.sync()
            if terminated or truncated:
                outcome = "reached the ball" if info["is_success"] else "did not reach the ball"
                print(f"episode {episode} (seed {args.seed + episode}): {outcome} "
                      f"after {info['time']:.2f} s")
                time.sleep(0.8 / args.speed)
                episode += 1
                obs, _ = env.reset(seed=args.seed + episode)
                viewer.sync()
            time.sleep(max(0.0, env.dt / args.speed - (time.perf_counter() - started)))
    env.close()


if __name__ == "__main__":
    main()
