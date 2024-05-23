import transformers
import llm_reconstruction_free as lrf
import os
from datasets import load_dataset_builder, get_dataset_split_names, load_dataset,concatenate_datasets
from argparse import ArgumentParser
import wandb
import numpy as np


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="rotten", choices=["rotten", "imdb"])
    parser.add_argument("--split", type=float, default=0.5)
    args = parser.parse_args()

    if args.dataset == "rotten":
        train_dataset, test_dataset = lrf.data.get_rotten()
    elif args.dataset == "imdb":
        train_dataset, test_dataset = lrf.data.get_imdb()


    def get_training_corpus():
        for i in range(0, len(train_dataset), 1000):
            yield train_dataset[i : i + 1000]["text"]

    tokenizer = lrf.tokenizer.train_identity(get_training_corpus())
    vocab = tokenizer.vocab
    print(type(tokenizer))
    print(tokenizer)
    def tokenization(example):
        return tokenizer(example["text"])
    print(train_dataset[0])
    dataset = train_dataset.map(tokenization, batched=True)
    print(dataset[0])


