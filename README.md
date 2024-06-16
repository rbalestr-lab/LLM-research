# LLM-reconstruction-free

## Installation

Install all the usual huggingface, Pytorch libraries. For ease, we provide the conda environment that was used for development into the file `environmnent.yml` (from running `conda env export > environment.yml`) which you can then use to install a new environment with

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

We leverage torchrun as in the following example

```
torchrun --nproc-per-node 8 supervised_finetuning.py --dataset rotten_tomatoes --lora-rank 8 --training-steps 500 --per-device-batch-size 4
```


# Datasets
- https://huggingface.co/datasets/tasksource/bigbench?row=1
- https://huggingface.co/datasets/google/civil_comments
- https://huggingface.co/datasets/amazon_polarity?row=11
- https://huggingface.co/datasets/yelp_polarity
- https://huggingface.co/datasets/fhamborg/news_sentiment_newsmtsc
- https://huggingface.co/datasets/nickmuchi/financial-classification
- https://huggingface.co/datasets/mwong/climate-evidence-related?row=48


# Models

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

