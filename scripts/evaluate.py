"""Evaluate the trained policy and the scripted baseline on the same seeded episodes.

    python scripts/evaluate.py                  # 200 episodes each
    python scripts/evaluate.py --episodes 20    # quick check

Writes results/eval.json (summary plus every episode) and results/eval.md (the README table).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from mujoco_ssl import GoToBallEnv
from mujoco_ssl.baseline import GoToBallController

Policy = Callable[[np.ndarray], np.ndarray]


def run_episode(env: GoToBallEnv, policy: Policy, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    total, steps, wall_steps = 0.0, 0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        steps += 1
        wall_steps += int(info["wall_contact"])
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "success": bool(info["is_success"]),
        "time_s": round(info["time"], 3),
        "return": round(total, 4),
        "out_of_field": bool(info["out_of_field"]),
        "timed_out": bool(truncated),
        "wall_contact_steps": wall_steps,
        "ball_displacement_m": round(info["ball_displacement"], 4),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a success rate."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def summarise(episodes: list[dict]) -> dict:
    n = len(episodes)
    solved = [e for e in episodes if e["success"]]
    times = np.array([e["time_s"] for e in solved])
    return {
        "episodes": n,
        "successes": len(solved),
        "success_rate": round(len(solved) / n, 4),
        "success_rate_ci95": wilson_interval(len(solved), n),
        "time_to_ball_s": {
            "mean": round(float(times.mean()), 3) if len(times) else None,
            "std": round(float(times.std(ddof=1)), 3) if len(times) > 1 else None,
            "median": round(float(np.median(times)), 3) if len(times) else None,
            "p90": round(float(np.percentile(times, 90)), 3) if len(times) else None,
        },
        "mean_return": round(float(np.mean([e["return"] for e in episodes])), 3),
        "out_of_field": sum(e["out_of_field"] for e in episodes),
        "timed_out": sum(e["timed_out"] for e in episodes),
        "episodes_with_wall_contact": sum(e["wall_contact_steps"] > 0 for e in episodes),
        "ball_moved_over_5cm": sum(e["ball_displacement_m"] > 0.05 for e in episodes),
    }


def paired_times(a: list[dict], b: list[dict], resamples: int = 10_000) -> dict:
    """Time difference a - b on the seeds both controllers solved, with a bootstrap CI."""
    both = [(x["time_s"], y["time_s"]) for x, y in zip(a, b) if x["success"] and y["success"]]
    diff = np.array([x - y for x, y in both])
    rng = np.random.default_rng(0)
    means = rng.choice(diff, size=(resamples, len(diff))).mean(axis=1)
    return {
        "seeds_solved_by_both": len(both),
        "mean_difference_s": round(float(diff.mean()), 3),
        "mean_difference_ci95_s": [round(float(q), 3) for q in np.percentile(means, [2.5, 97.5])],
        "first_faster_on": int((diff < 0).sum()),
        "second_faster_on": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def markdown_table(summaries: dict[str, dict]) -> str:
    rows = [
        "| Controller | Reached the ball | Success, 95% interval | Time to ball, mean ± sd (s) "
        "| Median (s) | 90th percentile (s) |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        t = s["time_to_ball_s"]
        lo, hi = s["success_rate_ci95"]
        rows.append(
            f"| {name} | {s['successes']} of {s['episodes']} | {100 * lo:.1f}–{100 * hi:.1f}% "
            f"| {t['mean']:.2f} ± {t['std']:.2f} | {t['median']:.2f} | {t['p90']:.2f} |"
        )
    return "\n".join(rows) + "\n"


def load_ppo(path: Path) -> Policy:
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/ppo_go_to_ball.zip"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("results/eval.json"))
    parser.add_argument("--table", type=Path, default=Path("results/eval.md"))
    args = parser.parse_args()

    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    controllers = {"PPO": load_ppo(args.model), "Scripted baseline": GoToBallController()}
    env = GoToBallEnv()
    episodes = {}
    for name, policy in controllers.items():
        episodes[name] = [run_episode(env, policy, seed) for seed in seeds]
    env.close()

    summaries = {name: summarise(eps) for name, eps in episodes.items()}
    result = {
        "env_id": "RoboCupSSL/GoToBall-v0",
        "model": args.model.as_posix(),
        "seeds": [seeds[0], seeds[-1]],
        "summary": summaries,
        "ppo_minus_baseline_time": paired_times(episodes["PPO"], episodes["Scripted baseline"]),
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    args.table.write_text(markdown_table(summaries))
    print(markdown_table(summaries))
    print(json.dumps(result["ppo_minus_baseline_time"], indent=1))


if __name__ == "__main__":
    main()
