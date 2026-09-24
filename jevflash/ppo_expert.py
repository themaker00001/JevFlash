#!/usr/bin/env python3
"""Load NanoJev's actual expert-generation ingredient: a frozen, pretrained
Sample Factory APPO policy (edbeeching/doom_basic_1111 on Hugging Face) for
ViZDoom's 'basic' scenario, and expose it as a simple per-frame action-probs
function.

This must run in a SEPARATE virtualenv (.venv-expert) from the rest of the
project: `sample-factory` pins an incompatible huggingface-hub version that
breaks `transformers`, which the training pipeline depends on. Never install
sample-factory into the main .venv.

Action index -> our label mapping (confirmed from sample-factory's
sf_examples/vizdoom/doom/doom_utils.py DoomSpec + doom_gym.py's
_convert_actions: Discrete(1+3) "idle, left, right, attack", and index>0
one-hots position (index-1) of the available_buttons list, which for
'basic' is declared MOVE_LEFT, MOVE_RIGHT, ATTACK in vizdoom's bundled
basic.cfg -- the same file jevflash/doom_env.py loads):
  0 -> noop, 1 -> left, 2 -> right, 3 -> shoot
"""
import sys
from pathlib import Path

import numpy as np
import torch

SAMPLE_FACTORY_SRC = "/tmp/sample-factory-src"
if SAMPLE_FACTORY_SRC not in sys.path:
    sys.path.insert(0, SAMPLE_FACTORY_SRC)

from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.enjoy import load_state_dict
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components, register_vizdoom_envs

ACTION_LABELS = ["noop", "left", "right", "shoot"]


class PPOExpert:
    def __init__(self, experiment="doom_basic_1111", train_dir="/tmp/sf_train_dir/vizdoom",
                 env="doom_basic", algo="APPO", device="cpu"):
        register_vizdoom_envs()
        register_vizdoom_components()
        argv = ["--algo", algo, "--env", env, "--experiment", experiment,
                "--train_dir", train_dir, "--device", device]
        parser, cfg = parse_sf_args(argv=argv, evaluation=True)
        cfg = parse_full_cfg(parser, argv)
        cfg = load_from_checkpoint(cfg)
        cfg.num_envs = 1
        from gymnasium.spaces import Box, Discrete
        obs_space = {"obs": Box(low=0, high=255, shape=(3, cfg.res_h, cfg.res_w), dtype=np.uint8)}
        import gymnasium as gym
        obs_space = gym.spaces.Dict(obs_space)
        action_space = Discrete(4)
        self.actor_critic = create_actor_critic(cfg, obs_space, action_space)
        self.actor_critic.eval()
        self.device = torch.device(device)
        self.actor_critic.model_to_device(self.device)
        load_state_dict(cfg, self.actor_critic, self.device)
        self.res_h, self.res_w = cfg.res_h, cfg.res_w
        self.rnn_size = get_rnn_size(cfg)
        self.reset()

    def reset(self):
        self.rnn_states = torch.zeros([1, self.rnn_size], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def action_probs(self, frame_hwc_uint8):
        """frame_hwc_uint8: numpy array (H, W, 3) uint8, ALREADY resized to
        (self.res_h, self.res_w) -- resize before calling this."""
        chw = np.transpose(frame_hwc_uint8, (2, 0, 1))
        obs = {"obs": torch.tensor(chw, dtype=torch.uint8, device=self.device).unsqueeze(0)}
        normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs)
        policy_outputs = self.actor_critic(normalized_obs, self.rnn_states, action_mask=None)
        self.rnn_states = policy_outputs["new_rnn_states"]
        action_distribution = self.actor_critic.action_distribution()
        probs = action_distribution.probs[0].cpu().tolist()
        return dict(zip(ACTION_LABELS, probs))
