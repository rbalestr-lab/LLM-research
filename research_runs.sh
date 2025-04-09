#!/bin/bash

set -e


# Part 1: Signing in to Wandb
# -------------------------------------------------------------------------

# Check if the user is already logged into WandB
if wandb status &>/dev/null; then
    echo "Already logged into WandB"
else
    echo "Not logged into WandB. Please log in:"
    wandb login --relogin
fi

for spur_type in "date" "countries"; do
    for seed in 5 10 15 20 25 30 35 40 45 50; do
        for location in "end" "beginning" "start"; do
            for lora_rank in  1 2 4 8 16 32; do
                for proportion in 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1; do
                    for token_proportion in 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do

                        echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

                        torchrun --nproc-per-node 1 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=$lora_rank ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
                    done
                done
            done
        done
    done
done

