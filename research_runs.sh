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

# for spur_type in "date"; do
#     for seed in 40; do
#         for location in "end" "beginning" "start"; do
#             for lora_rank in 16 32 64; do
#                 for proportion in  0 0.25 0.5 0.75 1; do
#                     for token_proportion in 0.05; do

#                         echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

#                         torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=$lora_rank ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
#                     done
#                 done
#             done
#         done
#     done
# done


# for spur_type in "date"; do
#     for seed in 45; do
#         for location in "beginning"; do
#             for lora_rank in  16; do
#                 for proportion in 0.75 1; do
#                     for token_proportion in 0.1; do

#                         echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

#                         torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=$lora_rank ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
#                     done
#                 done
#             done
#         done
#     done
# done

# for spur_type in "date"; do
#     for seed in 45; do
#         for location in "beginning"; do
#             for lora_rank in 32 64; do
#                 for proportion in 0 0.25 0.5 0.75 1; do
#                     for token_proportion in 0.1; do

#                         echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

#                         torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=$lora_rank ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
#                     done
#                 done
#             done
#         done
#     done
# done


for spur_type in "date"; do
    for seed in 40; do
        for location in "random"; do
            for lora_rank in 1 16 32 64; do
                for proportion in 0.5; do
                    for token_proportion in 0 0.1; do

                        echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

                        torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=$lora_rank ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
                    done
                done
            done
        done
    done
done

# for spur_type in "html"; do
#     for seed in 40; do
#         for location in "random"; do
#             for lora_rank in 64; do
#                 for proportion in 0.5; do
#                     for token_proportion in 0.1; do

#                         echo "Running with location=$location, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

#                         torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=$lora_rank ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
#                     done
#                 done
#             done
#         done
#     done
# done

# for spur_type in "exclamation_test"; do
#     for seed in 40; do
#         for location in "end"; do
#             for proportion in  0 0.25 0.5 0.75 1; do
#                 for token_proportion in 0 0.1; do
#                     echo "Running with location=$location, lora_rank=0, proportion=$proportion, seed=$seed, and token_proportion=$token_proportion"
#                     torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=0 ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
#                 done
#             done
#         done
#     done
# done


# for spur_type in "date"; do
#     for seed in 40; do
#         for location in "random"; do
#             for proportion in  0 0.25 0.5 0.75 1; do
#                 for token_proportion in 0 0.1; do
#                     echo "Running with location=$location, lora_rank=0, proportion=$proportion, seed=$seed, and token_proportion"
#                     torchrun --nproc-per-node 8 examples/supervised_finetuning.py --config-dir ./examples --config-name hydra.yaml ++params.spurious_proportion=$proportion ++params.spurious_token_proportion=$token_proportion ++params.spurious_location=$location ++params.spurious_test_proportion=$proportion ++params.spurious_test_token_proportion=$token_proportion ++params.spurious_test_location=$location ++params.lora_rank=0 ++params.spurious_type=$spur_type ++params.seed=$seed "$@"
#                 done
#             done
#         done
#     done
# done