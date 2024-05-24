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
