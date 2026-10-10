"""Render the README media: a GIF of the trained policy and a still of the field and robot.

    python scripts/render_media.py
    python scripts/render_media.py --episodes 5 --first-seed 1000

Each episode is played twice: once to plan a fixed camera that frames the whole approach
from in front of the robot's final pose, then again (identically, as the policy is
deterministic) to film it at 25 frames per second, every second control step, so the GIF
plays in real time. Offscreen rendering works as is on macOS; on a headless Linux machine
set MUJOCO_GL=egl.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mujoco_ssl import BALL, FIELD, ROBOT, GoToBallEnv

FPS = 25
HOLD_SECONDS = 1.0  # pause on the last frame of each episode
INK = (11, 11, 11)
PANEL = (255, 255, 255, 215)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


ELEVATION = -55.0  # degrees; negative looks down
FOVY = 45.0  # MuJoCo's default vertical field of view, degrees
MARGINS = (0.07, 0.14, 0.07, 0.08)  # left, top, right, bottom, as fractions of the frame


def project(points: np.ndarray, lookat: np.ndarray, distance: float, azimuth: float,
            width: int, height: int) -> np.ndarray:
    """Pixel coordinates of 3D points seen by a MuJoCo free camera."""
    az, el = math.radians(azimuth), math.radians(ELEVATION)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
    right = np.cross(forward, up)
    relative = points - (lookat - distance * forward)
    depth = relative @ forward
    focal = (height / 2) / math.tan(math.radians(FOVY / 2))
    u = width / 2 + focal * (relative @ right) / depth
    v = height / 2 - focal * (relative @ up) / depth
    return np.stack([u, v], axis=1)


def episode_camera(points: np.ndarray, azimuth: float, width: int,
                   height: int) -> mujoco.MjvCamera:
    """A fixed camera looking along `azimuth` (degrees) that keeps all `points` in frame.

    `points` are 3D. The look-at point and distance are adjusted a few times so the projected
    points fill the frame inside the margins.
    """
    az = math.radians(azimuth)
    ahead = np.array([math.cos(az), math.sin(az), 0.0])
    across = np.array([math.sin(az), -math.cos(az), 0.0])
    tilt = math.sin(math.radians(-ELEVATION))
    left, top, right, bottom = MARGINS
    box = np.array([(1 - left - right) * width, (1 - top - bottom) * height])
    box_centre = np.array([(left + (1 - left - right) / 2) * width,
                           (top + (1 - top - bottom) / 2) * height])
    lookat = np.array([points[:, 0].mean(), points[:, 1].mean(), 0.0])
    distance = 3.0
    for _ in range(30):
        pixels = project(points, lookat, distance, azimuth, width, height)
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        shift = (low + high) / 2 - box_centre
        metres_per_pixel = distance * 2 * math.tan(math.radians(FOVY / 2)) / height
        lookat = lookat + across * shift[0] * metres_per_pixel
        lookat = lookat - ahead * shift[1] * metres_per_pixel / tilt
        distance = max(distance * float(np.max((high - low) / box)) ** 0.8, 1.2)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = azimuth, ELEVATION, distance
    camera.lookat[:] = lookat
    return camera


def ball_pixels(frame: np.ndarray) -> int:
    """Count orange pixels, to check that the ball is in view."""
    r, g, b = (frame[..., i].astype(int) for i in range(3))
    return int(((r > 150) & (r - g > 50) & (b < 100)).sum())


def plan_camera(env: GoToBallEnv, policy, renderer: mujoco.Renderer,
                seed: int) -> tuple[mujoco.MjvCamera, float]:
    """Play the episode once without filming and choose a camera for it.

    The camera stands in front of the robot's final pose and to one side, so the robot faces
    it as it reaches the ball and never hides the ball. If the ball is still hidden in the last
    frame, the other side and then a head-on view are tried.
    """
    obs, _ = env.reset(seed=seed)
    points = [env.ball_position]
    while True:
        x, y, _ = env.robot_pose
        points += [(x, y, 0.0), (x, y, ROBOT.height)]
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            break
    x, y, _ = env.robot_pose
    points += [(x, y, 0.0), (x, y, ROBOT.height), env.ball_position]
    heading = math.degrees(env.robot_pose[2])
    for turn in (135.0, -135.0, 180.0):
        camera = episode_camera(np.array(points), heading + turn, renderer.width,
                                renderer.height)
        renderer.update_scene(env.data, camera=camera)
        if ball_pixels(renderer.render()) >= 30:
            break
    return camera, info["time"]


def caption(frame: np.ndarray, lines: list[str], font: ImageFont.ImageFont) -> Image.Image:
    """Draw a small white panel with text in the top-left corner."""
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad, gap = 10, 4
    heights = [draw.textbbox((0, 0), text, font=font)[3] for text in lines]
    width = max(draw.textbbox((0, 0), text, font=font)[2] for text in lines)
    box = (12, 12, 12 + width + 2 * pad, 12 + sum(heights) + gap * (len(lines) - 1) + 2 * pad)
    draw.rounded_rectangle(box, radius=6, fill=PANEL)
    y = box[1] + pad
    for text, height in zip(lines, heights):
        draw.text((box[0] + pad, y), text, font=font, fill=INK)
        y += height + gap
    return Image.alpha_composite(image, overlay).convert("RGB")


def record_episode(env: GoToBallEnv, policy, renderer: mujoco.Renderer, seed: int,
                   scale: int) -> tuple[list[np.ndarray], dict]:
    """Play the episode again (deterministically) and film it with the planned camera."""
    camera, duration = plan_camera(env, policy, renderer, seed)
    obs, info = env.reset(seed=seed)
    frames, step = [], 0
    every = round(1 / env.dt) // FPS
    while True:
        if step % every == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(downsample(renderer.render(), scale))
        obs, _, terminated, truncated, info = env.step(policy(obs))
        step += 1
        if terminated or truncated:
            renderer.update_scene(env.data, camera=camera)
            frames.append(downsample(renderer.render(), scale))
            if info["time"] != duration:
                raise RuntimeError("the replay differed from the first run")
            return frames, info


def downsample(frame: np.ndarray, scale: int) -> np.ndarray:
    """Average scale x scale pixel blocks: cheap supersampling for thin field lines."""
    if scale == 1:
        return frame
    h, w, c = frame.shape
    blocks = frame.reshape(h // scale, scale, w // scale, scale, c).astype(np.float32)
    return blocks.mean(axis=(1, 3)).round().astype(np.uint8)


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> int:
    """Quantise every frame to one shared palette (no dithering) and write the GIF."""
    sample = [frames[i] for i in np.linspace(0, len(frames) - 1, 24).astype(int)]
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(sample)))
    for i, frame in enumerate(sample):
        strip.paste(frame, (0, i * frames[0].height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    paletted = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(path, save_all=True, append_images=paletted[1:], duration=durations,
                     loop=0, optimize=False, disposal=1)
    return path.stat().st_size


def make_gif(args: argparse.Namespace, policy) -> None:
    env = GoToBallEnv()
    width, height, scale = args.width, args.height, 2
    renderer = mujoco.Renderer(env.model, height * scale, width * scale)
    font = load_font(15)
    frames, durations = [], []
    seeds = range(args.first_seed, args.first_seed + args.episodes)
    for number, seed in enumerate(seeds, start=1):
        raw, info = record_episode(env, policy, renderer, seed, scale)
        title = f"Trained PPO policy  ·  episode {number} of {args.episodes}"
        for frame in raw[:-1]:
            frames.append(caption(frame, [title], font))
            durations.append(1000 // FPS)
        result = (f"Reached the ball in {info['time']:.2f} s" if info["is_success"]
                  else "Did not reach the ball")
        frames.append(caption(raw[-1], [title, result], font))
        durations.append(int(HOLD_SECONDS * 1000))
        print(f"seed {seed}: success={info['is_success']} time={info['time']:.2f} s")
    renderer.close()
    env.close()
    size = save_gif(frames, durations, args.gif)
    print(f"wrote {args.gif}: {len(frames)} frames, {size / 1e6:.2f} MB")


def make_still(args: argparse.Namespace) -> None:
    """Top view of the whole field next to a close-up of the robot and the ball."""
    env = GoToBallEnv()
    x, y, yaw = -1.2, -0.9, 0.6
    reach = ROBOT.front_distance + BALL.radius + 0.045  # ball 4.5 cm in front of the face
    ball = (x + reach * math.cos(yaw), y + reach * math.sin(yaw))
    env.reset(seed=0, options={"robot_pose": (x, y, yaw), "ball_position": ball})
    scale = 2
    top_h = 520  # twice this must fit the 1080-pixel offscreen buffer
    top_w = round(top_h * (FIELD.length + 2 * FIELD.surface_margin)
                  / (FIELD.width + 2 * FIELD.surface_margin + 0.2))
    renderer = mujoco.Renderer(env.model, top_h * scale, top_w * scale)
    renderer.update_scene(env.data, camera="top")
    top = Image.fromarray(downsample(renderer.render(), scale))
    renderer.close()

    renderer = mujoco.Renderer(env.model, top_h * scale, top_h * scale)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (x + 0.06 * math.cos(yaw), y + 0.06 * math.sin(yaw), 0.06)
    camera.distance, camera.azimuth, camera.elevation = 0.55, math.degrees(yaw) + 120, -30.0
    renderer.update_scene(env.data, camera=camera)
    close = Image.fromarray(downsample(renderer.render(), scale))
    renderer.close()
    env.close()

    gap = 12
    still = Image.new("RGB", (top.width + gap + close.width, top_h), (255, 255, 255))
    still.paste(top, (0, 0))
    still.paste(close, (top.width + gap, 0))
    args.still.parent.mkdir(parents=True, exist_ok=True)
    still.save(args.still, optimize=True)
    print(f"wrote {args.still}: {still.size}, {args.still.stat().st_size / 1e3:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/ppo_go_to_ball.zip"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--gif", type=Path, default=Path("docs/images/go-to-ball-ppo.gif"))
    parser.add_argument("--still", type=Path, default=Path("docs/images/field-and-robot.png"))
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--skip-still", action="store_true")
    args = parser.parse_args()

    if not args.skip_still:
        make_still(args)
    if not args.skip_gif:
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")
        make_gif(args, lambda obs: model.predict(obs, deterministic=True)[0])


if __name__ == "__main__":
    main()
