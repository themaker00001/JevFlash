#!/usr/bin/env python3
"""Render a live ViZDoom 'basic' episode driven by a trained checkpoint's
own action choices (not the heuristic) to an animated GIF, so you can
actually watch what the model does instead of just reading success-rate
numbers.
"""
import argparse
from pathlib import Path

from PIL import Image

from doom_env import UnifiedDoomEnv
from play_doom import load_checkpoint, model_action
from train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/doom_model_episode.gif")
    p.add_argument("--scale", type=int, default=2)
    args = p.parse_args()

    device = pick_device(args.device)
    model, tokenizer, config = load_checkpoint(args.checkpoint_dir, device)
    env = UnifiedDoomEnv({"scenario": "basic", "max_steps": args.max_steps, "frame_skip": args.frame_skip})

    frames, actions_taken = [], []
    try:
        obs, info = env.reset(seed=args.seed)
        for step in range(args.max_steps):
            if not obs["candidates"]:
                break
            state = env._game.get_state()
            if state is not None:
                img = Image.fromarray(state.screen_buffer, "RGB")
                if args.scale > 1:
                    img = img.resize((img.width * args.scale, img.height * args.scale), Image.NEAREST)
                frames.append(img)
            action = model_action(model, tokenizer, obs, device)
            actions_taken.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        outcome = info["episode_metrics"]
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No frames captured")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=4 * args.frame_skip * (1000 // 35), loop=0)
    print(f"saved {len(frames)} frames to {args.output}")
    print(f"actions: {actions_taken}")
    print(f"outcome: success={outcome['success']} kills={outcome['kills']}")


if __name__ == "__main__":
    main()
