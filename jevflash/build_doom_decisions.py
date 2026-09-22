#!/usr/bin/env python3
"""Build a decision dataset from ViZDoom's 'basic' scenario using a simple
proportional aim-and-shoot heuristic as the gold policy.

Requires `pip install vizdoom==1.3.0` (see requirements-vizdoom.txt).
Each environment step (before the action is taken) becomes one training
question: given the current state, pick the best of {left, right, shoot,
noop}. The heuristic (not a learned expert) supplies the gold label:
strafe toward center-alignment with the target, then shoot once aligned.
"""
import argparse
import hashlib
import json
from pathlib import Path

from doom_env import UnifiedDoomEnv

CENTER_X = 160.0
TOLERANCE = 20.0


def target_x(state):
    labels = state["observed_history"][-1]["visible_labels"] if state["observed_history"] else []
    if not labels:
        return None
    box = labels[0]["bbox"]
    return box[0] + box[2] / 2


def heuristic_action(state):
    x = target_x(state)
    if x is None:
        return "noop"
    error = x - CENTER_X
    if abs(error) <= TOLERANCE:
        return "shoot"
    return "right" if error > 0 else "left"


def collect_episode(env, seed, family_id, split, max_steps):
    rows = []
    obs, info = env.reset(seed=seed)
    for step in range(max_steps):
        if not obs["candidates"]:
            break
        state = json.loads(obs["state"])
        gold = heuristic_action(state)
        state_id = f"doom:basic:{seed}:{step}"
        rows.append({
            "id": state_id, "state_id": state_id, "family_id": family_id, "split": split,
            "state": obs["state"],
            "questions": {
                "action": {"type": "choice", "instructions":
                           "Eliminate the monster before the task deadline. Pick the best action.",
                           "criteria": dict(obs["candidates"])},
            },
            "gold": {"action": gold},
            "metadata": {"source": "heuristic_proportional_aim_policy", "license": "CC0-1.0",
                         "source_group_id": f"doom:basic:{seed}", "seed": seed, "step": step},
        })
        obs, reward, done, truncated, info = env.step(gold)
        if done:
            break
    return rows, info.get("success", False)


def build(output_dir, seeds_by_split, max_steps, frame_skip):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = UnifiedDoomEnv({"scenario": "basic", "max_steps": max_steps, "frame_skip": frame_skip})
    manifest = {"schema_version": "jevflash-doom-v1", "scenario": "basic",
                "policy": "heuristic_proportional_aim", "splits": {}}
    try:
        for split, seeds in seeds_by_split.items():
            rows, successes = [], 0
            for seed in seeds:
                episode_rows, success = collect_episode(env, seed, "doom_basic_v1", split, max_steps)
                rows.extend(episode_rows)
                successes += int(success)
            payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
            (output_dir / f"{split}.jsonl").write_text(payload, encoding="utf-8")
            manifest["splits"][split] = {
                "episodes": len(seeds), "states": len(rows), "questions": len(rows),
                "episode_success_rate": successes / len(seeds) if seeds else None,
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
    finally:
        env.close()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="data/doom_basic_v1")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--train-episodes", type=int, default=16)
    p.add_argument("--dev-episodes", type=int, default=4)
    p.add_argument("--calibration-episodes", type=int, default=4)
    p.add_argument("--test-episodes", type=int, default=8)
    p.add_argument("--ood-episodes", type=int, default=4)
    args = p.parse_args()
    seeds_by_split = {
        "train": list(range(0, args.train_episodes)),
        "dev": list(range(1000, 1000 + args.dev_episodes)),
        "calibration": list(range(2000, 2000 + args.calibration_episodes)),
        "test": list(range(3000, 3000 + args.test_episodes)),
        "ood": list(range(4000, 4000 + args.ood_episodes)),
    }
    manifest = build(args.output_dir, seeds_by_split, args.max_steps, args.frame_skip)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
