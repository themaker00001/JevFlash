#!/usr/bin/env python3
"""Build a decision dataset from ViZDoom's 'basic' scenario using a simple
proportional aim-and-shoot heuristic as the gold policy.

Requires `pip install vizdoom==1.3.0` (see requirements-vizdoom.txt).
Each environment step (before the action is taken) becomes one training
question: given the current state, pick the best of {left, right, shoot,
noop}. The heuristic (not a learned expert) supplies the gold label:
strafe toward center-alignment with the target, then shoot once aligned.

--epsilon > 0 adds exploration during collection, following the technique
that made NanoJev's own ViZDoom distillation work (see docs/research):
the recorded LABEL for every state is always the heuristic's correct
action, but the action actually EXECUTED to advance the episode is the
heuristic's choice only (1-epsilon) of the time -- epsilon of the time a
uniformly random action is taken instead. This pushes the trajectory
slightly off the heuristic's own idealized path, and the still-correct
label for that off-path state is exactly the "how do I recover" signal a
purely on-policy heuristic rollout can never produce (since the heuristic
never makes a mistake to recover from). This is cheap here because the
heuristic itself is fast (no model inference during collection), unlike
DAgger against a slow, possibly-still-broken, learned model.
"""
import argparse
import hashlib
import json
import random
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


def collect_episode(env, seed, family_id, split, max_steps, epsilon, explore_rng):
    rows = []
    obs, info = env.reset(seed=seed)
    off_policy_steps = 0
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
        # The label above is always the heuristic's correct action. What we
        # actually DO to advance the episode can differ (see module
        # docstring) -- that's the whole point.
        if epsilon > 0 and explore_rng.random() < epsilon:
            executed = explore_rng.choice(list(obs["candidates"]))
            off_policy_steps += int(executed != gold)
        else:
            executed = gold
        obs, reward, done, truncated, info = env.step(executed)
        if done:
            break
    return rows, info.get("success", False), off_policy_steps


def build(output_dir, seeds_by_split, max_steps, frame_skip, epsilon=0.0, explore_seed=17):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = UnifiedDoomEnv({"scenario": "basic", "max_steps": max_steps, "frame_skip": frame_skip})
    explore_rng = random.Random(explore_seed)
    manifest = {"schema_version": "jevflash-doom-v1", "scenario": "basic",
                "policy": "heuristic_proportional_aim", "epsilon": epsilon, "splits": {}}
    try:
        for split, seeds in seeds_by_split.items():
            rows, successes, off_policy_total = [], 0, 0
            for seed in seeds:
                episode_rows, success, off_policy_steps = collect_episode(
                    env, seed, "doom_basic_v1", split, max_steps, epsilon, explore_rng)
                rows.extend(episode_rows)
                successes += int(success)
                off_policy_total += off_policy_steps
            payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
            (output_dir / f"{split}.jsonl").write_text(payload, encoding="utf-8")
            manifest["splits"][split] = {
                "episodes": len(seeds), "states": len(rows), "questions": len(rows),
                "episode_success_rate": successes / len(seeds) if seeds else None,
                "off_policy_steps": off_policy_total,
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
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="Probability of executing a random action instead of the heuristic's, "
                        "per step (label is always the heuristic's regardless). 0 = pure "
                        "on-policy heuristic rollout, matching the original v1/v2 data.")
    p.add_argument("--explore-seed", type=int, default=17)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Added to all split seed ranges, to generate a non-overlapping dataset "
                        "from the same episode counts (e.g. for a larger replacement train set).")
    args = p.parse_args()
    seeds_by_split = {
        "train": list(range(args.seed_offset, args.seed_offset + args.train_episodes)),
        "dev": list(range(args.seed_offset + 1000, args.seed_offset + 1000 + args.dev_episodes)),
        "calibration": list(range(args.seed_offset + 2000, args.seed_offset + 2000 + args.calibration_episodes)),
        "test": list(range(args.seed_offset + 3000, args.seed_offset + 3000 + args.test_episodes)),
        "ood": list(range(args.seed_offset + 4000, args.seed_offset + 4000 + args.ood_episodes)),
    }
    manifest = build(args.output_dir, seeds_by_split, args.max_steps, args.frame_skip,
                      epsilon=args.epsilon, explore_seed=args.explore_seed)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
