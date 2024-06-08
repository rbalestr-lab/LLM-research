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
  echo ${0} ${1} ${2} ${3} ${4} ${5} ${6} ${7}
  asdf
  torchrun --nproc-per-node 8 supervised_finetuning.py --dataset $dataset \
    --pretrained=$pretrained --training-steps $training_steps \
    --per-device-batch-size $per_device_batch_size --backbone $backbone \
    --from-gcs=$from_gcs --lora-rank=$lora_rank --freeze=$freeze \
    --vocab-size=$vocab_size --eval-steps=$eval_steps
  if [[ "1" != "${from_gcs}" ]]
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
#  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "" "0" "--pretrained"
#  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "4" "" "0" "--pretrained"
#  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "32" "" "0" "--pretrained"
  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "1" "0" "--pretrained"
#  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "" "4000" ""
}

training_steps=10
per_device_batch_size=1
backbone="Snowflake/snowflake-arctic-embed-xs"
from_gcs="0"
eval_steps=20
dataset=rotten_tomatoes

run_suite ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps}

#for dataset in "rotten_tomatoes" "sst2" "yelp_review_full" "imdb" "wiki_toxic" \
#  "toxigen" "bias_in_bios" "polarity" "emotion" "snli" "medical"
#do
#  echo "# wandb sync dataset ${dataset}"
#  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "--freeze" "0" "--pretrained"
#done

#dataset=rotten_tomatoes
#for backbone in "apple/OpenELM-270M" "apple/OpenELM-450M" "apple/OpenELM-1_1B" \
#  "apple/OpenELM-3B" "meta-llama/Meta-Llama-3-8B" "microsoft/phi-2" \
#  "Snowflake/snowflake-arctic-embed-xs" "Snowflake/snowflake-arctic-embed-s" \
#  "Snowflake/snowflake-arctic-embed-m" "Snowflake/snowflake-arctic-embed-l" \
#  "mistralai/Mistral-7B-v0.1"
#do
#  echo "# wandb sync backbone ${backbone}"
#  run_one ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps} "0" "--freeze" "0" "--pretrained"
#done
#
#backbone=apple/OpenELM-270M
#echo "# wandb sync run_suite"
#run_suite ${training_steps} ${per_device_batch_size} ${backbone} ${dataset} ${from_gcs} ${eval_steps}
