#!/bin/bash

# Usage: ./setup.sh

# Define the paths to the configuration files
CONFIG_FILE="imitation-in-homes/configs/env_vars/env_vars.yaml"

# Check if the config files exist
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Configuration file $CONFIG_FILE not found!"
    exit 1
fi

# Get the current directory
CURRENT_DIR="$(pwd)"

# Define the path to imitation-in-homes folder
IMI_HOME_DIR="$CURRENT_DIR/imitation-in-homes"

# Check if the imitation-in-homes folder exists
if [[ ! -d "$IMI_HOME_DIR" ]]; then
    echo "imitation-in-homes folder not found in the current directory!"
    exit 1
fi

# Update project_root path in env_vars.yaml
sed -i "s|^\(project_root: \).*|\1$IMI_HOME_DIR|" "$CONFIG_FILE"
echo "project_root updated to $IMI_HOME_DIR in $CONFIG_FILE"

# Download the checkpoint
wget https://pub-857f7aa7553b4b1e8396f9a2779e12ed.r2.dev/close.pt
wget https://pub-857f7aa7553b4b1e8396f9a2779e12ed.r2.dev/open.pt
wget https://pub-857f7aa7553b4b1e8396f9a2779e12ed.r2.dev/pick.pt
wget https://pub-857f7aa7553b4b1e8396f9a2779e12ed.r2.dev/pick_experimental.pt
mkdir imitation-in-homes/checkpoints
mv close.pt imitation-in-homes/checkpoints
mv open.pt imitation-in-homes/checkpoints
mv pick.pt imitation-in-homes/checkpoints
mv pick_experimental.pt imitation-in-homes/checkpoints