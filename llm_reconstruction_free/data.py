import transformers
from datasets import (
    load_dataset_builder,
    get_dataset_split_names,
    load_dataset,
    concatenate_datasets,
)

NAMES = [
    "civil_comments",
    "imdb",
    "rotten_tomatoes",
    "sst2",
    "yelp_review_full",
    "imdb",
]


def from_name(name):
    assert name in NAMES
    print(f"Loading {name}")
    splits = get_dataset_split_names(name)
    print("\t-splits:", splits)
    data = dict()
    for split in splits:
        data[split] = load_dataset(name, split=split)
        if "label" in data[split].column_names:
            data[split] = data[split].rename_column("label", "labels")
        assert "text" in data[split].column_names
        print(f"\t-{split}: {data[split].shape}")
    print("\t-columns:", data[split].column_names)
    return data


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


def get_yelp():
    train_dataset = load_dataset("yelp_review_full", split="train")
    test_dataset = load_dataset("yelp_review_full", split="test")
    return train_dataset, test_dataset


def get_sst2():
    train_dataset = load_dataset("sst2", split="train")
    test_dataset = load_dataset("sst2", split="test")
    return train_dataset, test_dataset


def get_civil():
    train_dataset = load_dataset("civil_comments", split="train")
    test_dataset = load_dataset("civil_comments", split="test")
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
