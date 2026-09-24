#!/usr/bin/env python3
"""Build a ViZDoom 'basic' decision dataset using NanoJev's actual expert --
a frozen, pretrained Sample Factory PPO policy (edbeeching/doom_basic_1111)
-- instead of our hand-coded heuristic, combined with the epsilon-greedy
exploration technique already validated in build_doom_decisions.py.

MUST run in .venv-expert (needs sample-factory + torch, incompatible with
the main .venv's transformers pin). Produces the exact same row schema as
build_doom_decisions.py, so it's a drop-in replacement dataset for
train.py / play_doom.py in the main venv.

Per step: capture the native 160x120 pixel frame (what the expert was
trained on), resize to 128x72, get the expert's action probabilities.
The training LABEL is always the expert's argmax action. The action
EXECUTED to advance the episode is the expert's argmax (1-epsilon) of the
time, uniformly random epsilon of the time -- manufacturing realistic
"recover from a small mistake" states, same as build_doom_decisions.py's
--epsilon but with a real learned expert instead of a 2-line heuristic.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from doom_env import UnifiedDoomEnv
from ppo_expert import PPOExpert


def collect_episode(env, expert, seed, family_id, split, max_steps, epsilon, explore_rng):
    rows = []
    expert.reset()
    obs, info = env.reset(seed=seed)
    off_policy_steps = 0
    for step in range(max_steps):
        if not obs["candidates"]:
            break
        raw_state = env._game.get_state()
        frame = np.array(Image.fromarray(raw_state.screen_buffer, "RGB").resize((128, 72), Image.NEAREST))
        probs = expert.action_probs(frame)
        gold = max(probs, key=probs.get)
        if gold not in obs["candidates"]:
            gold = "noop" if "noop" in obs["candidates"] else next(iter(obs["candidates"]))
        state_id = f"doom:ppo:{seed}:{step}"
        rows.append({
            "id": state_id, "state_id": state_id, "family_id": family_id, "split": split,
            "state": obs["state"],
            "questions": {
                "action": {"type": "choice", "instructions":
                           "Eliminate the monster before the task deadline. Pick the best action.",
                           "criteria": dict(obs["candidates"])},
            },
            "gold": {"action": gold},
            "metadata": {"source": "sample_factory_appo_doom_basic_1111_expert", "license": "CC0-1.0",
                         "source_group_id": f"doom:ppo:{seed}", "seed": seed, "step": step,
                         "expert_probs": probs},
        })
        if epsilon > 0 and explore_rng.random() < epsilon:
            executed = explore_rng.choice(list(obs["candidates"]))
            off_policy_steps += int(executed != gold)
        else:
            executed = gold
        obs, reward, done, truncated, info = env.step(executed)
        if done:
            break
    return rows, info.get("success", False), off_policy_steps


def build(output_dir, seeds_by_split, max_steps, frame_skip, epsilon, explore_seed, expert):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = UnifiedDoomEnv({"scenario": "basic", "max_steps": max_steps, "frame_skip": frame_skip,
                         "screen_resolution": "RES_160X120"})
    explore_rng = random.Random(explore_seed)
    manifest = {"schema_version": "jevflash-doom-v1", "scenario": "basic",
                "policy": "sample_factory_appo_doom_basic_1111", "epsilon": epsilon, "splits": {}}
    try:
        for split, seeds in seeds_by_split.items():
            rows, successes, off_policy_total = [], 0, 0
            for seed in seeds:
                episode_rows, success, off_policy_steps = collect_episode(
                    env, expert, seed, "doom_basic_ppo_v1", split, max_steps, epsilon, explore_rng)
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
    p.add_argument("--output-dir", default="data/doom_basic_ppo_v1")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--train-episodes", type=int, default=200)
    p.add_argument("--dev-episodes", type=int, default=8)
    p.add_argument("--calibration-episodes", type=int, default=4)
    p.add_argument("--test-episodes", type=int, default=16)
    p.add_argument("--ood-episodes", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--explore-seed", type=int, default=17)
    p.add_argument("--seed-offset", type=int, default=20000)
    args = p.parse_args()
    seeds_by_split = {
        "train": list(range(args.seed_offset, args.seed_offset + args.train_episodes)),
        "dev": list(range(args.seed_offset + 1000, args.seed_offset + 1000 + args.dev_episodes)),
        "calibration": list(range(args.seed_offset + 2000, args.seed_offset + 2000 + args.calibration_episodes)),
        "test": list(range(args.seed_offset + 3000, args.seed_offset + 3000 + args.test_episodes)),
        "ood": list(range(args.seed_offset + 4000, args.seed_offset + 4000 + args.ood_episodes)),
    }
    expert = PPOExpert()
    manifest = build(args.output_dir, seeds_by_split, args.max_steps, args.frame_skip,
                      args.epsilon, args.explore_seed, expert)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
