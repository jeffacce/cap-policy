#!/bin/bash

include_task="Door_Opening"
BATCH_SIZE=176

now_date=$(date '+%Y-%m-%d')
now_time=$(date '+%H-%M-%S')

RVQ_PATH="checkpoints/${now_date}/${include_task}-${include_env}-${now_time}"

wandb=true  # Set the wandb configuration here

cd imitation-in-homes

WANDB_MODE=disabled python train.py --config-name=train_rvq.yaml include_task=$include_task model_path=$RVQ_PATH
sleep 10

WANDB_MODE=$( [ "$wandb" = true ] && echo "online" || echo "disabled" ) accelerate launch --config_file configs/accelerate/accel_cfg.yaml train.py --config-name=train_vqbet.yaml include_task=$include_task vqvae_load_dir=$RVQ_PATH/checkpoint.pt batch_size=$BATCH_SIZE