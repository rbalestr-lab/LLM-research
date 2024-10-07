function run_one() {
  training_steps=${1}
  per_device_batch_size=${2}
  backbone=${3}
  dataset=${4}
  from_gcs=${5}
  eval_steps=${6}
  lora_rank=${7}
  freeze=${8}
  vocab_size=${9}
  pretrained=${10}
  echo ${0} ${1} ${2} ${3} ${4} ${5} ${6} lora_rank=${7} freeze=${8} \
    vocab_size=${9} pretrained=${10}
  torchrun --nproc-per-node 8 supervised_finetuning.py --dataset $dataset \
    --pretrained=$pretrained --training-steps $training_steps \
    --per-device-batch-size $per_device_batch_size --backbone $backbone \
    --from-gcs=$from_gcs --lora-rank=$lora_rank --freeze=$freeze \
    --vocab-size=$vocab_size --eval-steps=$eval_steps
  if [[ "none" != "${from_gcs}" ]]
  then
    gsutil -m cp -r /workdir/xcloud/wandb/* ${from_gcs}/wandb
    rm -rf /workdir/xcloud/wandb/*
  fi
}

function run_suite() {
  training_steps=${1}
  per_device_batch_size=${2}
  backbone=${3}
  dataset=${4}
  from_gcs=${5}
  eval_steps=${6}
  # run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "0" "0" "1"
  # run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "4" "0" "0" "1"
  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "32" "0" "0" "1"
  # run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "1" "0" "1"
  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "0" "32000" "0"
}

training_steps=2000
per_device_batch_size=1
backbone="microsoft/phi-2"
from_gcs=gs://haih-e1ff76673758
eval_steps=50
dataset=rotten_tomatoes

# MODELS = [
#     "apple/OpenELM-270M",
#     "apple/OpenELM-450M",
#     "apple/OpenELM-1_1B",
#     "apple/OpenELM-3B",
#     "meta-llama/Meta-Llama-3-8B",
#     "microsoft/phi-2",
#     "Snowflake/snowflake-arctic-embed-xs",
#     "Snowflake/snowflake-arctic-embed-s",
#     "Snowflake/snowflake-arctic-embed-m",
#     "Snowflake/snowflake-arctic-embed-l",
#     "Qwen/Qwen2-0.5B",
#     "Qwen/Qwen2-1.5B",
#     "Qwen/Qwen2-7B",
#     "mistralai/Mistral-7B-v0.1",
#     "mistralai/Mistral-7B-v0.3",
#     "google/gemma-2b",
#     "google/gemma-7b",
# ]

# for model_batch in "apple/OpenELM-270M:8" "apple/OpenELM-450M:8" \
#   "apple/OpenELM-1_1B:4" "apple/OpenELM-3B:2"
for model_batch in "meta-llama/Meta-Llama-3-8B:1" "microsoft/phi-2:2" \
  "Snowflake/snowflake-arctic-embed-xs:8" \
  "Snowflake/snowflake-arctic-embed-s:8" \
  "Snowflake/snowflake-arctic-embed-m:8" "Snowflake/snowflake-arctic-embed-l:8"
# for model_batch in "Qwen/Qwen2-0.5B:8" "Qwen/Qwen2-1.5B:4" "Qwen/Qwen2-7B:1"
# for model_batch in "mistralai/Mistral-7B-v0.1:1" "mistralai/Mistral-7B-v0.3:1"
# for model_batch in "google/gemma-2b:2" "google/gemma-7b:1"
do
  backbone=$(echo ${model_batch} | cut -d ':' -f 1)
  per_device_batch_size=$(echo ${model_batch} | cut -d ':' -f 2)
  echo "model ${backbone} per_device_batch_size ${per_device_batch_size}"
  for dataset in "rotten_tomatoes" "sst2" "yelp_review_full" "imdb" \
    "wiki_toxic" "toxigen" "bias_in_bios" "polarity" "emotion" "snli" "medical"
  do
    echo "# wandb sync model ${backbone} dataset ${dataset}"
    run_suite ${training_steps} ${per_device_batch_size} ${backbone} \
      ${dataset} ${from_gcs} ${eval_steps}
  done
done

# for model_batch in "Snowflake/snowflake-arctic-embed-l:8" \
#   "apple/OpenELM-1_1B:4" "microsoft/phi-2:2"
# do
#   backbone=$(echo ${model_batch} | cut -d ':' -f 1)
#   per_device_batch_size=$(echo ${model_batch} | cut -d ':' -f 2)
#   echo "model ${backbone} per_device_batch_size ${per_device_batch_size}"
#   for dataset_step in "imdb:16000" "imbd:32000" "imdb:64000" \
#     "yelp_review_full:64000" "yelp_review_full:128000" "yelp_review_full:256000"
#   do
#     dataset=$(echo ${dataset_step} | cut -d ':' -f 1)
#     training_steps=$(echo ${dataset_step} | cut -d ':' -f 2)
#     echo "dataset ${dataset} training_steps ${training_steps}"
#     echo "# wandb sync model ${backbone} dataset ${dataset} training_steps ${training_steps}"
#     run_suite ${training_steps} ${per_device_batch_size} ${backbone} \
#       ${dataset} ${from_gcs} ${eval_steps}
#   done
# done

# dataset=sst2
# # for backbone in "apple/OpenELM-270M" "apple/OpenELM-450M" "apple/OpenELM-1_1B" \
# #   "apple/OpenELM-3B" "meta-llama/Meta-Llama-3-8B" "microsoft/phi-2" \
# #   "Snowflake/snowflake-arctic-embed-xs" "Snowflake/snowflake-arctic-embed-s" \
# #   "Snowflake/snowflake-arctic-embed-m" "Snowflake/snowflake-arctic-embed-l" \
# #   "mistralai/Mistral-7B-v0.1"
# for backbone in "Snowflake/snowflake-arctic-embed-xs" \
#   "Snowflake/snowflake-arctic-embed-s" "Snowflake/snowflake-arctic-embed-m" \
#   "Snowflake/snowflake-arctic-embed-l"
# do
#   echo "# wandb sync backbone ${backbone}"
#   run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "1" "0" "1"
# done

# backbone=apple/OpenELM-270M
# echo "# wandb sync run_suite"
# run_suite ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps}
