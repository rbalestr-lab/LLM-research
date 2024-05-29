import transformers
from datasets import (
    load_dataset_builder,
    get_dataset_split_names,
    load_dataset,
    load_from_disk,
    concatenate_datasets,
)

from . import gcs

NAMES = [
    "civil_comments",
    "rotten_tomatoes",
    "sst2",
    "yelp_review_full",
    "imdb",
    "wiki_toxic",
    "toxigen",
]


def from_name(name: str, from_gcs: str = None):
    assert name in NAMES
    if name == "wiki_toxic":
        name = "OxAISH-AL-LLM/wiki_toxic"
    elif name == "toxigen":
        name = "toxigen/toxigen-data"
    print(f"Loading {name}")
    local_cache = None
    if from_gcs:
        local_cache = gcs.local_copy(from_gcs, "datasets", name)
        splits = get_dataset_split_names(local_cache)
    else:
        splits = get_dataset_split_names(name)
    print("\t-splits:", splits)
    if from_gcs:
        data = load_from_disk(local_cache)
    else:
        data = dict()
        for split in splits:
            data[split] = load_dataset(name, split=split)
    for split in splits:
        if "label" in data[split].column_names:
            data[split] = data[split].rename_column("label", "labels")
        if name == "OxAISH-AL-LLM/wiki_toxic":
            assert "comment_text" in data[split].column_names
            data[split] = data[split].rename_column("comment_text", "text")
        elif name == "toxigen/toxigen-data":
            assert "toxicity_human" in data[split].column_names
            data[split] = data[split].rename_column("toxicity_human", "labels")
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
