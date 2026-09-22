#!/usr/bin/env python3
"""DAgger-style data aggregation for the ViZDoom 'basic' decision dataset.

The original dataset (build_doom_decisions.py) only contains states visited
by the heuristic's own clean trajectory, so a model trained on it never
sees what to do after making a mistake -- once it drifts off that narrow
path during live play, errors compound with no corrective signal in
training. This script fixes that at the data level: it lets the CURRENT
trained checkpoint drive live episodes (so the states visited reflect the
learner's own, possibly imperfect, behavior), and labels every state it
visits with the heuristic's action -- not the model's -- as the gold
target. Aggregating this with the original data teaches the model how to
recover, which is the whole point of DAgger (Ross et al., 2011).
"""
import argparse
import hashlib
import json
from pathlib import Path

from build_doom_decisions import heuristic_action
from doom_env import UnifiedDoomEnv
from play_doom import load_checkpoint, model_action
from train import pick_device


def collect_episode(env, seed, model, tokenizer, device, family_id, split, max_steps):
    rows = []
    obs, info = env.reset(seed=seed)
    for step in range(max_steps):
        if not obs["candidates"]:
            break
        state = json.loads(obs["state"])
        gold = heuristic_action(state)
        state_id = f"doom:dagger:{seed}:{step}"
        rows.append({
            "id": state_id, "state_id": state_id, "family_id": family_id, "split": split,
            "state": obs["state"],
            "questions": {
                "action": {"type": "choice", "instructions":
                           "Eliminate the monster before the task deadline. Pick the best action.",
                           "criteria": dict(obs["candidates"])},
            },
            "gold": {"action": gold},
            "metadata": {"source": "dagger_learner_rollout_heuristic_label", "license": "CC0-1.0",
                         "source_group_id": f"doom:dagger:{seed}", "seed": seed, "step": step},
        })
        # Step with the MODEL's own choice (not the heuristic's) -- this is
        # what makes the visited states reflect the learner's actual
        # trajectory, including its mistakes.
        action = model_action(model, tokenizer, obs, device)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    return rows, info.get("success", False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output", default="data/doom_basic_v1/train_dagger.jsonl")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--seed-offset", type=int, default=6000, help="Must not overlap build_doom_decisions.py's seed ranges")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = pick_device(args.device)
    model, tokenizer, config = load_checkpoint(args.checkpoint_dir, device)
    env = UnifiedDoomEnv({"scenario": "basic", "max_steps": args.max_steps, "frame_skip": args.frame_skip})
    seeds = list(range(args.seed_offset, args.seed_offset + args.episodes))
    rows, successes = [], 0
    try:
        for seed in seeds:
            episode_rows, success = collect_episode(env, seed, model, tokenizer, device,
                                                      "doom_basic_dagger_v1", "train", args.max_steps)
            rows.extend(episode_rows)
            successes += int(success)
    finally:
        env.close()

    payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(payload, encoding="utf-8")
    print(json.dumps({"output": args.output, "episodes": len(seeds), "states": len(rows),
                      "learner_episode_success_rate": successes / len(seeds),
                      "sha256": hashlib.sha256(payload.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
