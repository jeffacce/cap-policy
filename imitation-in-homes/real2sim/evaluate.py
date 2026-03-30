import os
import time
import torch
import gymnasium as gym
import numpy as np
import multiprocessing
from tqdm import tqdm
import json
import itertools
from itertools import cycle
import random

from .stripped_policy import StrippedPolicy
from .utils import download_and_extract_zip, get_objects, set_global_seeds

def setup_environment(num_envs, num_episodes, render_mode, objects, condition, seed):
    env_args = {
        "scene_path": "real2sim/scene.xml",
        "render_mode": render_mode,
        "secondary_objects": objects,
        "add_secondary_object": condition is not None
    }

    vectorize = num_envs > 1

    if vectorize:

        flat_objects = [obj for obj in objects]
        expanded_objects = list(itertools.islice(cycle(flat_objects), num_episodes))
        np.random.shuffle(expanded_objects)
        env_objects = [expanded_objects[i::num_envs] for i in range(num_envs)]

        env = gym.vector.AsyncVectorEnv(
            [
                (
                    lambda idx=i: gym.make(
                        "RUM-Pick-v0",
                        objects=env_objects[idx],
                        seed=seed + idx,
                        **env_args,
                    )
                )
                for i in range(num_envs)
            ]
        )

    else:
        env = gym.make("RUM-Pick-v0", objects=objects, seed=seed, **env_args)
        env_objects = [objects]

    return env, env_objects


def download_objects_if_needed():
    objects_path = "real2sim/assets/objects"
    if not os.path.exists(objects_path):
        print("Downloading Objects...")
        download_and_extract_zip(
            gdrive_link="https://drive.google.com/file/d/1AGtrkGBUPYdCdcO4e2raZxQpOGlIbwdn",
            output_dir="real2sim/assets",
        )
    return objects_path


def eval_sim(
    model,
    loss_fn,
    buffer_size,
    objects_set_path="real2sim/objects_sets/all_objects.json",
    num_envs=1,
    num_episodes=1000,
    render=True,
    condition=None,
    seed: int = 0,
    logs_path=None,
    ):
    set_global_seeds(seed)
    device = next(model.parameters()).device
    policy = StrippedPolicy(
        model=model,
        loss_fn=loss_fn,
        buffer_size=buffer_size,
        device=device,
        condition=condition,
    )

    objects_path = download_objects_if_needed()
    available_objects = get_objects(objects_path)
    available_objects = {obj[0]: obj[1] for obj in available_objects}

    with open(objects_set_path, "r") as file:
        objects_set = np.array(json.load(file))
        np.random.shuffle(objects_set)

    objects = []
    for requested_object in objects_set:
        if available_objects.get(requested_object) is not None:
            objects.append([requested_object, available_objects[requested_object]])
        else:
            print(
                f"Requested object {requested_object} not found in available objects at {objects_path}."
            )

    if num_envs > multiprocessing.cpu_count():
            print(
                f"Warning: num_envs ({num_envs}) is greater than the number of available CPU cores ({multiprocessing.cpu_count()}), switching to {multiprocessing.cpu_count()}."
            )
            num_envs = multiprocessing.cpu_count()
        
    if num_episodes < num_envs:
        print(
            f"Warning: num_episodes ({num_episodes}) is less than num_envs ({num_envs}), switching to {num_episodes}."
        )
        num_envs = num_episodes 
        
    env, env_objects = setup_environment(
        num_envs,
        num_episodes,
        render_mode="human_goal" if condition is not None else "human",
        objects=objects,
        condition=condition,
        seed=seed,
    )

    episodes_per_env = [len(env_objects[i]) for i in range(num_envs)]
    env_episode_count = np.zeros(num_envs, dtype=int)

    success = 0
    pbar = tqdm(
        total=num_episodes,
        desc="Sim Eval",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_noinv_fmt}] "
            "{postfix}"
        ),
    )
    vectorize = num_envs > 1
    logs = []

    if vectorize:
        obs, _ = env.reset()
        while np.any(env_episode_count < episodes_per_env):
            obs_rgb = obs["rgb"]
            goal_2d = obs["object_2d_position"]
            goal_3d = obs["object_3d_position"]

            if condition is None:
                actions = policy.forward(obs_rgb)
            elif condition == "2d":
                actions = policy.forward(obs_rgb, goal_2d)
            elif condition == "3d":
                actions = policy.forward(obs_rgb, goal_3d)
            else:
                raise NotImplementedError(
                    f"Condition {condition} not implemented. Use '2d' or '3d'."
                )

            obs, rewards, terminated, truncated, infos = env.step(actions)
            dones = np.logical_or(terminated, truncated)
            done_indices = np.where(dones)[0]
            policy.reset(done_indices)

            for idx in done_indices:
                if env_episode_count[idx] < len(env_objects[idx]):
                    logs.append(
                    {   
                        "object_name": infos["object_name"][idx],
                        "num_steps": infos["step_count"][idx],
                        "success": int(rewards[idx] > 0),
                    }
                    )
                    env_episode_count[idx] += 1
                    pbar.update(1)
                    if rewards[idx] > 0:
                        success += 1
                    pbar.set_postfix(
                        {
                            "success_rate": f"{(success / pbar.n) * 100:.1f}%",
                            "last_object": infos["object_name"][idx],
                        }
                    )

            if render:
                env.render()

        pbar.close()
        return {"success_rate": success / num_episodes, "logs": logs}

    else:
        episode = 0
        observation, _ = env.reset()
        while episode < num_episodes:
            obs_rgb = observation["rgb"]
            goal_2d = observation["object_2d_position"]

            if condition is None:
                action = policy.forward(np.expand_dims(obs_rgb, axis=0))[0]
            elif condition == "2d":
                action = policy.forward(
                    np.expand_dims(obs_rgb, axis=0), np.expand_dims(goal_2d, axis=0)
                )[0]
            elif condition == "3d":
                goal_3d = observation["object_3d_position"]
                action = policy.forward(
                    np.expand_dims(obs_rgb, axis=0), np.expand_dims(goal_3d, axis=0)
                )[0]
            else:
                raise NotImplementedError(
                    f"Condition {condition} not implemented. Use '2d' or '3d'."
                )

            observation, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

            if terminated or truncated:
                policy.reset([0])
                success += int(reward > 0)
                episode += 1
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "success_rate:": f"{(success / episode) * 100:.1f}%",
                        "last_object:": f"{info['object_name']}",
                    }
                )
                logs.append(
                    {
                        "object_name": info["object_name"],
                        "num_steps": info["step_count"],
                        "success": int(reward > 0),
                    }
                )
                observation, _ = env.reset()

        pbar.close()

        if logs_path is not None:
            os.makedirs(os.path.dirname(logs_path), exist_ok=True)
            with open(logs_path, "w") as f:
                f.write("object_name,success,num_steps\n")
                for e in logs:
                    f.write(f"{e['object_name']},{e['success']},{e['num_steps']}\n")

        return {"success_rate": success / episode, "logs": logs}
