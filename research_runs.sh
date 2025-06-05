#!/bin/bash

set -e

export HF_DATASETS_CACHE=/opt/dlami/nvme/hf_cache
export TRANSFORMERS_CACHE=/opt/dlami/nvme/hf_cache
export HF_HOME=/opt/dlami/nvme/hf_cache

# Part 1: Signing in to Wandb
# -------------------------------------------------------------------------

# Check if the user is already logged into WandB
if wandb status &>/dev/null; then
    echo "Already logged into WandB"
else
    echo "Not logged into WandB. Please log in:"
    wandb login --relogin
fi

#45 50

# for the bigger model have done: 0.05
# Need to do 0, 0.1 (token proportion)

# have done seed: 40
# need to do 45, 50

# for spur_type in "date"; do
#     for seed in 40; do
#         for location in "random"; do
#             for lora_rank in 1 16 32 64; do
#                 for proportion in 0.5; do
#                     for token_proportion in 0; do

#                         echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

#                         torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=$lora_rank ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
#                     done
#                 done
#             done
#         done
#     done
# done
# 
# MAKE SURE TO TURN BACK THE DATE RANGE

# for spur_type in "date"; do
#     for seed in 40; do
#         for location in "random"; do
#             for lora_rank in 64; do
#                 for proportion in 0; do
#                     for token_proportion in 0; do

#                         echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

#                         torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.lora_rank=$lora_rank ++params.seed=$seed ++params.use_spurious=False "$@"
#                     done
#                 done
#             done
#         done
#     done
# done

for spur_type in "date"; do
    for seed in 40; do
        for location in "end" "beginning" "random"; do
            for lora_rank in 1 16 32 64; do
                for proportion in 0.5; do
                    for token_proportion in 0.1; do
                        for training_steps in 5000

                            echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

                            torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=$lora_rank ++params.spurious_type=$spur_type ++params.seed=$seed ++params.training_steps=$training_steps "$@"

                        done
                    done
                done
            done
        done
    done
done