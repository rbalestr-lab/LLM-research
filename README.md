# LLM-reconstruction-free

## Installation

Install all the usual huggingface, Pytorch libraries. For ease, we provide the conda environment that was used for development into the file `environmnent.yml` (from running `conda env export --no-builds | grep -v "prefix" > environment.yml`) which you can then use to install a new environment with

```
conda env create -n ENVNAME --file ENV.yml
```

Before running any script, make sure to set the following environment variables, e.g., by adding them to your `.bashrc`:

```
export OMP_NUM_THREADS=10
export TOKENIZERS_PARALLELISM=false
```

The first is system depending and is used by `torchrun` when runnning the script, adjust it based on your needs. 
The second is needed as we use tokenizers which doesn't play well with other multiprocessing going in when training


## Run the scripts

We leverage torchrun as in the following example:

```
torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --lora-rank 8 --training-steps 500 --per-device-batch-size 4
```

Furthermore, we leverage hydra (https://hydra.cc/) to schedule runs through submitit_slurm. The following is an example of 
what a call could look like which takes advantage of the ./run_model.sh script:

```
./run_model.sh ++params.training_steps=500 ++params.per_device_batch_size=4 ++params.use_spurious=True,False ++params.backbone=Snowflake/snowflake-arctic-embed-xs ++params.freeze=0 ++params.pretrained=0  ++params.dataset=common_sense ++params.spurious_proportion=0.75 ++params.spurious_token_proportion=0.1 ++params.spurious_location=random ++params.spurious_label=1 ++params.spurious_test_label=1 ++params.spurious_test_proportion=0.75 ++params.spurious_test_token_proportion=0.1 ++params.spurious_test_location=random ++hydra.launcher.partition=cs-all-gcondo
```


# Datasets
Here is a list of the datasets currently supported. They are ready to be used out of the box. To learn more reference [data.py](./llm_research/data.py).

- https://huggingface.co/datasets/tasksource/bigbench?row=1
- https://huggingface.co/datasets/google/civil_comments
- https://huggingface.co/datasets/amazon_polarity?row=11
- https://huggingface.co/datasets/yelp_polarity
- https://huggingface.co/datasets/fhamborg/news_sentiment_newsmtsc
- https://huggingface.co/datasets/nickmuchi/financial-classification
- https://huggingface.co/datasets/mwong/climate-evidence-related?row=48
- https://huggingface.co/datasets/tau/commonsense_qa
- https://huggingface.co/datasets/stanfordnlp/imdb
- https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
- https://huggingface.co/datasets/ehovy/race
- https://huggingface.co/datasets/LabHC/bias_in_bios


# Models
We currently support many different models. This is a list of the models that are currently supported out of the box. To learn more reference the [__init__.py](./llm_research/__init__.py) file for the [llm_research](./llm_research/) directory.


- apple/OpenELM-270M
- apple/OpenELM-450M
- apple/OpenELM-1_1B
- apple/OpenELM-3B
- meta-llama/Meta-Llama-3-8B
- microsoft/phi-2
- Snowflake/snowflake-arctic-embed-xs
- Snowflake/snowflake-arctic-embed-s
- Snowflake/snowflake-arctic-embed-m
- Snowflake/snowflake-arctic-embed-l
- Qwen/Qwen2-0.5B
- Qwen/Qwen2-1.5B
- Qwen/Qwen2-7B
- mistralai/Mistral-7B-v0.1
- mistralai/Mistral-7B-v0.3
- google/gemma-2b
- google/gemma-7b

  
For OpenELM we have to run the following to get the submodules (no need for the user to do that again, you should just clone the repo with `git clone --recurse-submodules`)
```
export GIT_LFS_SKIP_SMUDGE=1 

git submodule add https://huggingface.co/apple/OpenELM-270M
git mv OpenELM-270M OpenELM_270M

git submodule add https://huggingface.co/apple/OpenELM-450M
git mv OpenELM-450M OpenELM_450M

git submodule add https://huggingface.co/apple/OpenELM-1_1B
git mv OpenELM-1_1B OpenELM_1_1B

git submodule add https://huggingface.co/apple/OpenELM-3B
git mv OpenELM-3B OpenELM_3B




Default:

attention_dropout
num_hidden_layers
num_attention_heads
vocab_size
hidden_size
max_position_embeddings


Qwen:
defaults

OpenELM:

max_position_embeddings -> max_context_length
num_hidden_layers -> num_transformer_layers + update the num_kv_heads and num_query_heads and ffn_multipliers
hidden_size -> model_dim
attention_dropout -> None
num_attention_heads -> num_kv_heads and num_query_heads (lists)
vocab_size -> vocab_size

arctic

attention_dropout -> attention_probs_dropout_prob + hidden_dropout_prob
hidden_size -> hidden_size + intermediate_size
max_position_embeddings -> max_position_embeddings
num_hidden_layers -> num_hidden_layers
num_attention_heads -> num_attention_heads
vocab_size -> vocab_size


Llama3

attention_dropout -> attention_dropout
hidden_size -> hidden_size
intermediate_size -> intermediate_size
max_position_embeddings -> max_position_embeddings
num_attention_heads -> num_attention_heads
num_hidden_layers -> num_hidden_layers
num_key_value_heads -> num_key_value_heads
vocab_size -> vocab_size

