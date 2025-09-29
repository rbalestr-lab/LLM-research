#!/bin/bash

# Part 1: Setting up and activating Conda Environment
# -------------------------------------------------------------------------
# Set the conda environment name"
ENV_NAME="iblora_env"

# Ensure that Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda and try again."
    exit 1
fi

# Ensure the Conda environment exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Activating existing Conda environment: $ENV_NAME"
    # Activate the Conda environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

else
    echo "Creating Conda environment: $ENV_NAME"
    # Create the conda environment and activate it
    eval "$(conda shell.bash hook)"
    conda env create -n $ENV_NAME --file environment.yml
    conda activate "$ENV_NAME"
fi

# Print out the current environment
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Part 2: Signing in to Wandb
# -------------------------------------------------------------------------

# Check if the user is already logged into WandB
if wandb status &>/dev/null; then
    echo "Already logged into WandB"
else
    echo "Not logged into WandB. Please log in:"
    wandb login --relogin
fi



# Part 3: Running the Training
# -------------------------------------------------------------------------
# Run the training script
#
# Acceptable parameters to change and their default values:
# dataset: rotten_tomatoes
# seed: 50
# per_device_batch_size: 8
# freeze: 0
# pretrained: 0
# use_spurious: False
# backbone: apple/OpenELM-450M
# lora_rank: 0
# training_steps: 200
# batch_size: 64
# pretrained_tokenizer:  None
# weight_decay: 1e-5
# learning_rate: 1e-4
# dropout: 0
# mixup: 0
# vocab_size: None
# max_length: 1024
# label_smoothing: 0
# from_gcs: none
# eval_steps: 20
# mixture: 0
# lora0: 0
# superlinear: none
# scaling_gamma: 0
# use_dora: 0
# spurious_location: random
# total_parameters: 0
# training_parameters: 0

echo "Starting training..."
# python examples/supervised_finetuning.py --multirun --config-dir ./examples --config-name hydra "$@"
python examples/attention_viz.py --multirun --config-dir ./examples --config-name hydra "$@"