import transformers
from datasets import (
    load_dataset_builder,
    get_dataset_split_names,
    load_dataset,
    concatenate_datasets,
)


def get_pretraining_dataset():

    print("splits for rotten", get_dataset_split_names("rotten_tomatoes"))
    print("splits for imdb", get_dataset_split_names("imdb"))
    print("splits for snli", get_dataset_split_names("snli"))

    rotten_tomatoes = load_dataset("rotten_tomatoes", split="train")
    imdb = load_dataset("imdb", split="train")
    snli = load_dataset("snli", split="train")

    # for pre-training
    snli = snli.rename_column("premise", "text")
    snli = snli.remove_columns("hypothesis")

    # for pre training
    snli = snli.remove_columns("label")
    imdb = imdb.remove_columns("label")
    rotten_tomatoes = rotten_tomatoes.remove_columns("label")
    train_dataset = concatenate_datasets([rotten_tomatoes, imdb, snli])
    return train_dataset


def get_rotten():
    train_dataset = load_dataset("rotten_tomatoes", split="train")
    test_dataset = load_dataset("rotten_tomatoes", split="test")
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")
    return train_dataset, test_dataset


def get_snli():
    train_dataset = load_dataset("snli", split="train")
    test_dataset = load_dataset("snli", split="test")
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")
    return train_dataset, test_dataset

def get_imdb():
    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")
    return train_dataset, test_dataset
