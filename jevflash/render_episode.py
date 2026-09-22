#!/usr/bin/env python3
"""Render a ViZDoom 'basic' episode to an animated GIF, using the same
proportional aim-heuristic as build_doom_decisions.py. Runs fully headless
(no window) but captures the actual rendered screen buffer, unlike
doom_env.py which only extracts text (health/ammo/label boxes).
"""
import argparse
from pathlib import Path

import vizdoom as vzd
from PIL import Image

CENTER_X = 160.0
TOLERANCE = 20.0


def heuristic_action(state):
    labels = [l for l in state.labels if l.object_name != "DoomPlayer"]
    if not labels:
        return [False, False, False]
    label = labels[0]
    x = label.x + label.width / 2
    error = x - CENTER_X
    if abs(error) <= TOLERANCE:
        return [False, False, True]  # shoot
    return [False, True, False] if error > 0 else [True, False, False]  # right : left


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--output", default="results/doom_episode.gif")
    p.add_argument("--scale", type=int, default=2, help="Integer upscale factor for visibility")
    args = p.parse_args()

    game = vzd.DoomGame()
    scenarios = Path(vzd.scenarios_path)
    game.load_config(str(scenarios / "basic.cfg"))
    game.set_doom_scenario_path(str(scenarios / "basic.wad"))
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.init()
    game.set_seed(args.seed)
    game.new_episode()

    frames = []
    for step in range(args.max_steps):
        if game.is_episode_finished():
            break
        state = game.get_state()
        if state is None:
            break
        img = Image.fromarray(state.screen_buffer, "RGB")
        if args.scale > 1:
            img = img.resize((img.width * args.scale, img.height * args.scale), Image.NEAREST)
        frames.append(img)
        action = heuristic_action(state)
        game.make_action(action, args.frame_skip)

    killcount = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    game.close()

    if not frames:
        raise RuntimeError("No frames captured")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=4 * args.frame_skip * (1000 // 35),
                   loop=0)
    print(f"saved {len(frames)} frames to {args.output}, killcount={killcount}")


if __name__ == "__main__":
    main()
