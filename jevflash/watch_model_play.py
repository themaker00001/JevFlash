#!/usr/bin/env python3
"""Open a real, visible ViZDoom window and let a trained checkpoint play
live episodes, paced so a human can actually watch it -- not a recording.
"""
import argparse
import time

from doom_env import UnifiedDoomEnv
from play_doom import load_checkpoint, model_action
from train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=9000)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--step-delay", type=float, default=0.35, help="Seconds to pause per decision, for watchability")
    args = p.parse_args()

    device = pick_device(args.device)
    model, tokenizer, config = load_checkpoint(args.checkpoint_dir, device)
    env = UnifiedDoomEnv({"scenario": "basic", "max_steps": args.max_steps, "frame_skip": args.frame_skip,
                         "window_visible": True})
    try:
        for i in range(args.episodes):
            seed = args.seed_offset + i
            obs, info = env.reset(seed=seed)
            print(f"=== episode {i+1}/{args.episodes} (seed {seed}) ===", flush=True)
            steps = 0
            while obs["candidates"]:
                action = model_action(model, tokenizer, obs, device)
                print(f"  step {steps}: {action}", flush=True)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                time.sleep(args.step_delay)
                if done:
                    outcome = info["episode_metrics"]
                    print(f"  -> success={outcome['success']} steps={steps}", flush=True)
                    break
            time.sleep(1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
