#!/bin/bash

# Part 1: Setting up and activating Conda Environment
# -------------------------------------------------------------------------
# Set the conda environment name"
ENV_NAME = "llm"

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
echo "Starting training..."
python examples/supervised_finetuning.py
