torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --pretrained --training-steps 100 --per-device-batch-size 4

torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --pretrained --freeze --training-steps 100 --per-device-batch-size 4

torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --pretrained --lora-rank 4 --training-steps 100 --per-device-batch-size 4

