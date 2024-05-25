training_steps=5000
per_device_batch_size=4

# pretrained model that will fully fine-tune
#torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --pretrained --training-steps $training_steps --per-device-batch-size $per_device_batch_size

# pretrained model that is frozen and fine-tuned with LORA
#torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --pretrained --training-steps $training_steps --per-device-batch-size $per_device_batch_size --lora-rank 4
#torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --pretrained --training-steps $training_steps --per-device-batch-size $per_device_batch_size --lora-rank 32

# pretrained model that is frozen and not fine-tuned (baseline)
#torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --pretrained --freeze --training-steps $training_steps --per-device-batch-size $per_device_batch_size

# randomly initialized model that we train from scratch, including tokenizer
#torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --training-steps $training_steps --per-device-batch-size $per_device_batch_size --vocab-size 4000

#torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --training-steps $training_steps --per-device-batch-size $per_device_batch_size --vocab-size 4000 --label-smoothing 0.1

torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --training-steps $training_steps --per-device-batch-size $per_device_batch_size --vocab-size 4000 --weight-decay 0.0001

torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --training-steps $training_steps --per-device-batch-size $per_device_batch_size --vocab-size 4000 --dropout 0.5

torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --training-steps $training_steps --per-device-batch-size $per_device_batch_size --vocab-size 4000 --mixup 1
