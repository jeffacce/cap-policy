#!/bin/bash

include_task="Door_Opening"

wandb=true  # Set the wandb configuration here

cd imitation-in-homes

WANDB_MODE=$( [ "$wandb" = true ] && echo "online" || echo "disabled" ) python train.py --config-name=train_vqbet include_task=$include_task vqvae_load_dir=null device=cpu