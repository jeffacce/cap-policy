#!/bin/bash

include_task="Door_Opening"

wandb=true  # Set the wandb configuration here

cd imitation-in-homes

WANDB_MODE=$( [ "$wandb" = true ] && echo "online" || echo "disabled" ) HYDRA_FULL_ERROR=1 python train.py --config-name=train_rvq include_task=$include_task device=cpu