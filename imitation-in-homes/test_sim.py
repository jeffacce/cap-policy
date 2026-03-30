import os
import time
import torch
import gymnasium as gym
import numpy as np
import multiprocessing
import argparse
from hydra import initialize, compose
from datetime import datetime

from real2sim import eval_sim
from real2sim.utils import init_model_loss_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        help="Path to the model weights (.pt file)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global RNG seed for reproducibility",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        required=False,
        default=100,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--condition",
        default=None,
        type=str,
        required=False,
        choices=["2d", "3d"],
        help="Conditioning type for the model",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        required=False,
        default=1,
        help="Number of environments to evaluate in parallel",
    )
    parser.add_argument(
        "--objects_set_path",
        type=str,
        required=False,
        default="real2sim/objects_sets/lite_objects.json",
        help="Path to the JSON file containing the object sets",
    )
    parser.add_argument(
        "--encoder", 
        default=None,
        type=str,
        required=False,
        help="Encoder used in the model",
    )
    parser.add_argument(
        "--larger_gpt",
        action="store_true",
        help="Use larger GPT for the model",
    )
    args = parser.parse_args()

    with initialize(config_path="configs"):
        if args.encoder is not None:
            cfg = compose(config_name="run_vqbet", overrides=[f"model={args.encoder}"])
        else:
            cfg = compose(config_name="run_vqbet")

    if args.model_path is None:
        if args.condition is None:
            args.model_path = "general-pick-up-v1.pt"
        elif args.condition == "2d":
            args.model_path = "general-pick-up-goal-2-full-rvq-5k-demos.pt"
        elif args.condition == "3d":
            args.model_path = "general-pick-up-goal-3-5k-demos.pt"

    if args.condition == "2d":
        cfg.goal_dim = 2
        cfg.gpt_input_dim = 512
    elif args.condition == "3d":
        cfg.goal_dim = 3
        cfg.gpt_input_dim = 512
    else:
        cfg.goal_dim = 0
        cfg.gpt_input_dim = 256
    
    if args.larger_gpt:
        cfg.loss_fn.gpt_model.config.n_layer = 8
        cfg.loss_fn.gpt_model.config.n_head = 8
        cfg.loss_fn.gpt_model.config.n_embd = 256
        
    cfg.model_weight_pth = args.model_path
    cfg.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model, loss_fn = init_model_loss_fn(cfg)
    path = f"real2sim/logs/logs_testsim_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    results = eval_sim(
        model=model,
        loss_fn=loss_fn,
        objects_set_path=args.objects_set_path,
        buffer_size=cfg.image_buffer_size,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        render=args.render,
        condition=args.condition,
        seed=args.seed,
        logs_path=path,
    )

    print(f"{results['success_rate']*100:.2f} success rate across {args.num_episodes} episodes")    
    print(f"Saved logs to {path}")
