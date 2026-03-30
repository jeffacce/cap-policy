from typing import Tuple

import cv2
import hydra
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
import os
import sys

from utils.trajectory_vis import visualize_trajectory

DEFAULT_TASK_CHECKPOINTS = {
    "open": "checkpoints/open.pt",
    "close": "checkpoints/close.pt",
    "pick": "checkpoints/pick.pt",
}

class WrapperPolicy(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def step(self, data, *args, **kwargs):
        model_out = self.model(data)
        return self.loss_fn.step(data, model_out)

    def reset(self):
        pass

def print_help():
    """Print comprehensive help information for run.py"""
    print("Imitation-in-Homes Run Script - Configuration Help")
    print("=================================================")
    print("Usage: python run.py [OPTIONS] [HYDRA_OVERRIDES]")
    print()
    print("Available Options:")
    print("  -h, --help    Show this help message")
    print()
    print("Hydra Configuration Overrides:")
    print("  You can override any configuration parameter using the format: parameter=value")
    print()
    print("Available Configuration Parameters:")
    print()
    print("1. Checkpoint and Model Paths:")
    print("   model_weight_pth=<path>        Path to model checkpoint (local or HuggingFace)")
    print("                                  Examples: '/path/to/checkpoint.pt'")
    print("                                  Examples: 'hf://username/repo/checkpoint.pt'")
    print("                                  Default: checkpoints/open.pt, close.pt, or pick.pt by task")
    print("   checkpoint_path=<path>         Alternative checkpoint path for open-loop models")
    print("   vqvae_load_dir=<path>          Path for VQVAE model loading")
    print("                                  Default: null")
    print()
    print("2. Task and Device Configuration:")
    print("   task=<task_name>               Task to run")
    print("                                  Options: open, close, pick")
    print("                                  Default: pick")
    print("   device=<device>                Device to run on (cpu, cuda)")
    print("                                  Default: cpu (run.yaml), cuda (run_vqbet.yaml)")
    print("   run_offline=<bool>             Run offline evaluation vs. robot control")
    print("                                  Default: false")
    print()
    print("3. Network Configuration:")
    print("   network.host=<ip>              Host IP address")
    print("                                  Default: '127.0.0.1'")
    print("   network.remote=<ip>            Remote IP address")
    print("                                  Default: '127.0.0.1'")
    print("   network.camera_port=<port>     Camera port")
    print("                                  Default: 32922")
    print("   network.action_port=<port>     Action port")
    print("                                  Default: 8081")
    print("   network.flag_port=<port>       Flag port")
    print("                                  Default: 2828")
    print("   network.pose_port=<port>       Pose port")
    print("                                  Default: 32932")
    print()
    print("4. Robot Parameters:")
    print("   robot_params.h=<float>         Base height parameter from starting point")
    print("                                  Default: 0.6 (run.yaml), 0.0 (run_vqbet.yaml)")
    print("   robot_params.max_h=<float>     Maximum height deviation from base height")
    print("                                  Default: 0.06 (run.yaml), 0.10 (run_vqbet.yaml)")
    print("   robot_params.max_base=<float>  Maximum base movement from starting point")
    print("                                  Default: 0.08")
    print("   robot_params.abs_gripper=<bool> Absolute gripper mode")
    print("                                  Default: True")
    print("   robot_params.rot_unit=<str>    Rotation unit")
    print("                                  Default: 'axis'")
    print()
    print("5. Model and Training Parameters:")
    print("   image_buffer_size=<int>        Number of images in buffer")
    print("                                  Default: 1 (run.yaml), 3 (run_vqbet.yaml)")
    print("   temperature=<float>            Temperature for sampling")
    print("                                  Default: 0.000001")
    print("   sequentially_select=<bool>     Sequential selection mode")
    print("                                  Default: true")
    print("   vqvae_n_embed=<int>            VQVAE number of embeddings")
    print("                                  Default: 16")
    print("   goal_dim=<int>                 Goal dimension")
    print("                                  Default: 3")
    print("   gpt_input_dim=<int>            GPT input dimension")
    print("                                  Default: 512")
    print()
    print("6. Data and Visualization:")
    print("   use_depth=<bool>               Whether to use depth data")
    print("                                  Default: false")
    print("   stream_depth=<bool>            Whether to stream depth")
    print("                                  Default: True (run_vqbet.yaml)")
    print("   use_pose=<bool>                Whether to use pose data")
    print("                                  Default: True (run_vqbet.yaml)")
    print("   image_save_dir=<path>          Directory to save images")
    print("                                  Default: ${env_vars.project_root}/robot_images")
    print("   goal_conditional=<bool>        Whether model is goal conditional")
    print("                                  Default: false")
    print()
    print("7. UI and Logging:")
    print("   use_ui=<bool>                  Whether to use UI")
    print("                                  Default: false")
    print("   use_vlm=<bool>                 Whether to use VLM")
    print("                                  Default: false")
    print("   wandb.entity=<str>             Weights & Biases entity")
    print("   wandb.project=<str>            Weights & Biases project")
    print("                                  Default: 'imitation-in-homes'")
    print("   wandb.id=<str>                 Weights & Biases run ID")
    print("   wandb.save_code=<bool>         Whether to save code to W&B")
    print("                                  Default: true")
    print()
    print("Examples:")
    print("  python run.py                                    # Use default configuration")
    print("  python run.py task=drawer_opening               # Change task")
    print("  python run.py device=cuda                       # Use GPU")
    print("  python run.py model_weight_pth=/path/to/model.pt # Use local checkpoint")
    print("  python run.py model_weight_pth=hf://user/repo/model.pt # Use HuggingFace checkpoint")
    print("  python run.py network.host=192.168.1.100       # Change host IP")
    print("  python run.py robot_params.max_h=0.15           # Change robot height")
    print("  python run.py run_offline=true                  # Run offline evaluation")
    print("  python run.py task=bag_pick_up device=cuda network.host=192.168.1.100  # Multiple overrides")
    print()
    # print("Predefined Tasks Available:")
    # print("  - door_opening: Door opening task")
    # print("  - drawer_opening: Drawer opening task")
    # print("  - reorientation: Object reorientation task")
    # print("  - bag_pick_up: Bag pick up task")
    # print("  - tissue_pick_up: Tissue pick up task")
    print()
    print("Note: All configuration overrides use Hydra's override syntax.")
    print("      You can also use Hydra's built-in help: python run.py --help")
    print("      For Hydra's configuration explorer: python run.py --cfg job")

def _resolve_model_weight_pth(cfg):
    """
    Resolve model weight path: config first, then default for open/close/pick.
    If config path is set but file doesn't exist (local path), fall back to default.
    """
    model_weight_pth = cfg.get("model_weight_pth")
    task_name = cfg.get("task")
    default_pth = DEFAULT_TASK_CHECKPOINTS.get(task_name)

    if model_weight_pth is not None:
        if model_weight_pth.startswith("hf://"):
            return model_weight_pth
        if os.path.exists(model_weight_pth):
            return model_weight_pth
        print(f"Config path '{model_weight_pth}' not found, falling back to default for task '{task_name}'")
        if default_pth is not None:
            return default_pth
        raise FileNotFoundError(
            f"model_weight_pth '{model_weight_pth}' not found and no default for task '{task_name}'"
        )

    if default_pth is None:
        raise ValueError(
            f"No model_weight_pth provided and no default checkpoint configured "
            f"for task '{task_name}'. Please set model_weight_pth."
        )
    print(f"Using default checkpoint for task '{task_name}': {default_pth}")
    return default_pth


def load_checkpoint_from_path(checkpoint_path, device):
    """
    Load checkpoint from local path or HuggingFace URL.
    
    Args:
        checkpoint_path: Local file path or HuggingFace URL (hf://username/repo/filename)
        device: Device to load the checkpoint on
    
    Returns:
        Loaded checkpoint dictionary
    """
   
    if checkpoint_path.startswith("hf://"):
        # Handle HuggingFace URL
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required for loading from HuggingFace. Install with: pip install huggingface_hub")
        
        # Parse hf://username/repo/filename format
        parts = checkpoint_path[5:].split("/")  # Remove "hf://" and split
        if len(parts) < 3:
            raise ValueError(f"Invalid HuggingFace URL format: {checkpoint_path}. Expected: hf://username/repo/filename")
        
        username = parts[0]
        repo = parts[1]
        filename = "/".join(parts[2:])  # Handle nested paths
        
        repo_id = f"{username}/{repo}"
        print(f"Downloading checkpoint from HuggingFace: {repo_id}/{filename}")
        
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return torch.load(local_path, map_location=device, weights_only=False)
    else:
        # Handle local file path
        return torch.load(checkpoint_path, map_location=device, weights_only=False)

def _init_model(cfg):
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.device)

    model_weight_pth = _resolve_model_weight_pth(cfg)
    checkpoint = load_checkpoint_from_path(model_weight_pth, cfg.device)
    model.load_state_dict(checkpoint["model"])
    return model

def _init_model_loss(cfg):
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.device)

    model_weight_pth = _resolve_model_weight_pth(cfg)
    checkpoint = load_checkpoint_from_path(model_weight_pth, cfg.device)

    model.load_state_dict(checkpoint["model"])
    loss_fn = hydra.utils.instantiate(cfg.loss_fn, model=model)
    loss_fn.load_state_dict(checkpoint["loss_fn"])
    loss_fn = loss_fn.to(cfg.device)
    
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_parameters = sum(p.numel() for p in loss_fn.parameters() if p.requires_grad)
    # print model params in millions with %.2f
    print(f"Model parameters: {model_parameters / 1e6:.2f}M")
    print(f"Loss parameters: {loss_parameters / 1e6:.2f}M")
        
    policy = WrapperPolicy(model, loss_fn)
    return policy

def _init_open_loop(cfg):
    model = hydra.utils.instantiate(cfg.model)
    if cfg.checkpoint_path is not None:
        checkpoint = load_checkpoint_from_path(cfg.checkpoint_path, cfg.device)
        model.encoder.load_state_dict(checkpoint["model"])
    model = model.to(cfg.device)
    train_dataloader = _setup_dataloaders(cfg)
    model.set_dataset(train_dataloader)
    return model

def _init_simple_replay(cfg):
    model = hydra.utils.instantiate(cfg.model)
    return model

def _setup_dataloaders(cfg) -> Tuple[DataLoader]:
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    train_sampler = hydra.utils.instantiate(cfg.sampler, dataset=train_dataset)
    train_batch_sampler = hydra.utils.instantiate(
        cfg.batch_sampler, dataset=train_dataset
    )
    train_dataloader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        sampler=train_sampler,
        batch_sampler=train_batch_sampler,
    )
    return train_dataloader

def run(cfg: OmegaConf, init_model=_init_model):
    model = init_model(cfg)
    if cfg["run_offline"] is True:
        test_dataset = hydra.utils.instantiate(cfg.dataset.test)
        visualize_trajectory(
            model,
            test_dataset,
            cfg["device"],
            cfg["image_buffer_size"],
            goal_conditional=cfg["goal_conditional"],
        )

    else:
        # Lazy loading so we can run offline eval without the robot set up.
        from robot.controller import Controller

        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        controller = Controller(cfg=dict_cfg)
        controller.setup_model(model)
        controller.run()


@hydra.main(config_path="configs", config_name="run_vqbet", version_base="1.2")
def main(cfg: OmegaConf):
    if "simplereplay" in str.lower(cfg.model["_target_"]):
        run(cfg, init_model=_init_simple_replay)
    elif "replay" in str.lower(cfg.model["_target_"]):
        run(cfg, init_model=_init_open_loop)
    elif cfg.get("loss_fn") is None:
        run(cfg)
    else:
        run(cfg, init_model=_init_model_loss)

if __name__ == "__main__":
    # Check if user wants help
    if len(sys.argv) > 1 and any(arg in ['-h', '--help', 'help'] for arg in sys.argv):
        print_help()
        sys.exit(0)
    
    main()
