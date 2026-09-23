#!/usr/bin/env python3
"""Drive a live ViZDoom 'basic' episode using a trained JevFlash checkpoint,
picking the argmax action each step (closed-loop, not scoring a static
dataset). Also runs a uniform-random baseline for comparison.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from doom_env import UnifiedDoomEnv
from train import DecisionModel, pick_device, autocast_ctx


def load_checkpoint(checkpoint_dir, device):
    root = Path(checkpoint_dir)
    config = json.loads((root / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    body_config = AutoConfig.from_pretrained(str(root / "backbone_config"))
    body_config.use_cache = False
    body = AutoModel.from_config(body_config, attn_implementation="sdpa").float()
    model = DecisionModel(body, config["set_head"])
    weights = load_file(str(root / "best.safetensors"), device="cpu")
    model.load_state_dict(weights, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, tokenizer, config


def build_example(obs, tokenizer, max_length=512):
    state = obs["state"]
    ids = list(obs["candidates"])
    # Shuffle to match training (see train.py's load_examples): the live
    # env always presents candidates in the same fixed order, so without
    # this, any residual positional bias would still show up here even
    # after training on shuffled data.
    random.shuffle(ids)
    texts = [f"{key}: {obs['candidates'][key]}" for key in ids]
    segments = [f"State:\n{state}\n",
                "Question type: choice\nQuestion:\nEliminate the monster before the task deadline. "
                "Pick the best action.\n"]
    prefix = sum([tokenizer.encode(t, add_special_tokens=False) for t in segments], [])
    leaves = [prefix + tokenizer.encode(f"Candidate:\n{t}\nDecision:", add_special_tokens=False)
              + [tokenizer.eos_token_id] for t in texts]
    if max(map(len, leaves)) > max_length:
        raise ValueError("state/question too long for max_length")
    return {"type": "choice", "candidate_ids": ids, "leaf_tokens": leaves}


@torch.no_grad()
def model_action(model, tokenizer, obs, device):
    example = build_example(obs, tokenizer)
    with autocast_ctx(device):
        logits, _ = model([example], tokenizer.pad_token_id)
    k = len(example["candidate_ids"])
    probs = logits[0, :k].float().softmax(-1).tolist()
    best = max(range(k), key=probs.__getitem__)
    return example["candidate_ids"][best]


def run_episode(env, seed, action_fn):
    obs, info = env.reset(seed=seed)
    steps = 0
    while obs["candidates"]:
        action = action_fn(obs)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        if done:
            return {"seed": seed, "success": info["success"], "steps": steps,
                    "outcome": info["episode_metrics"]["outcome"] if "outcome" in info["episode_metrics"] else info.get("terminal_reason")}
    return {"seed": seed, "success": False, "steps": steps, "outcome": "no_candidates"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-offset", type=int, default=9000)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/doom_basic_eval.json")
    p.add_argument("--include-random-baseline", action="store_true")
    args = p.parse_args()

    device = pick_device(args.device)
    model, tokenizer, config = load_checkpoint(args.checkpoint_dir, device)

    env = UnifiedDoomEnv({"scenario": "basic", "max_steps": args.max_steps, "frame_skip": args.frame_skip})
    seeds = list(range(args.seed_offset, args.seed_offset + args.episodes))
    try:
        model_runs = [run_episode(env, s, lambda obs: model_action(model, tokenizer, obs, device)) for s in seeds]
        random_runs = None
        if args.include_random_baseline:
            import random as _random
            rng = _random.Random(0)
            random_runs = [run_episode(env, s, lambda obs: rng.choice(list(obs["candidates"]))) for s in seeds]
    finally:
        env.close()

    def summarize(runs):
        successes = [r for r in runs if r["success"]]
        return {
            "episodes": len(runs), "success_rate": len(successes) / len(runs),
            "mean_steps_to_kill": statistics.mean(r["steps"] for r in successes) if successes else None,
            "median_steps_to_kill": statistics.median(r["steps"] for r in successes) if successes else None,
            "outcomes": {o: sum(1 for r in runs if r["outcome"] == o) for o in set(r["outcome"] for r in runs)},
            "runs": runs,
        }

    result = {"checkpoint": args.checkpoint_dir, "backbone": config.get("model"),
              "device": device.type, "model": summarize(model_runs)}
    if random_runs is not None:
        result["random_baseline"] = summarize(random_runs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "model" or True} , indent=2)[:2000])


if __name__ == "__main__":
    main()
