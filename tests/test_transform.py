import pytest
from datasets import load_dataset
from spurious_corr.generators import SpuriousDateGenerator, SpuriousFileItemGenerator
from spurious_corr.modifiers import ItemInjection
from spurious_corr.transform import spurious_transform
import os


@pytest.fixture(scope="module")
def imdb_dataset():
    """
    Load IMDB dataset directly from Hugging Face.
    """
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].select(range(200))  # subsample for test speed
    return train_dataset


@pytest.fixture(scope="module")
def color_list():
    """
    Load colors.txt from local data directory.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    color_file = os.path.join(data_dir, "colors.txt")

    assert os.path.exists(color_file), f"colors.txt not found at {color_file}"

    with open(color_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def test_spurious_transform_proportion_multiple(imdb_dataset, color_list):
    label_to_modify = 1
    modifier = ItemInjection.from_list(color_list, token_proportion=0.5, seed=23)
    originals = [ex for ex in imdb_dataset]

    for text_proportion in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        transformed = spurious_transform(
            label_to_modify=label_to_modify,
            dataset=imdb_dataset,
            modifier=modifier,
            text_proportion=text_proportion,
            seed=42,
        )

        modified_count = sum(
            1
            for orig, mod in zip(originals, transformed)
            if orig["label"] == label_to_modify and orig["text"] != mod["text"]
        )

        total_to_modify = sum(1 for ex in originals if ex["label"] == label_to_modify)
        expected = round(total_to_modify * text_proportion)

        print(
            f"[text_proportion={text_proportion}] Modified: {modified_count} / Expected: {expected}"
        )
        assert (
            modified_count == expected
        ), f"Expected {expected}, but got {modified_count} at proportion {text_proportion}"


def test_spurious_transform_reproducible(imdb_dataset):
    date_generator_1 = SpuriousDateGenerator(seed=19, with_replacement=False)
    modifier_1 = ItemInjection.from_function(
        date_generator_1, token_proportion=0.5, seed=19
    )

    date_generator_2 = SpuriousDateGenerator(seed=19, with_replacement=False)
    modifier_2 = ItemInjection.from_function(
        date_generator_2, token_proportion=0.5, seed=19
    )

    transformed1 = spurious_transform(0, imdb_dataset, modifier_1, 0.3, seed=19)
    transformed2 = spurious_transform(0, imdb_dataset, modifier_2, 0.3, seed=19)

    texts1 = [ex["text"] for ex in transformed1]
    texts2 = [ex["text"] for ex in transformed2]

    assert texts1 == texts2, "Expected reproducible output with same seed"


def test_spurious_transform_different_seeds(imdb_dataset):
    date_generator_1 = SpuriousDateGenerator(seed=19, with_replacement=False)
    modifier_1 = ItemInjection.from_function(
        date_generator_1, token_proportion=0.5, seed=19
    )

    date_generator_2 = SpuriousDateGenerator(seed=19, with_replacement=False)
    modifier_2 = ItemInjection.from_function(
        date_generator_2, token_proportion=0.5, seed=19
    )

    transformed1 = spurious_transform(0, imdb_dataset, modifier_1, 0.3, seed=19)
    transformed2 = spurious_transform(0, imdb_dataset, modifier_2, 0.3, seed=20)

    texts1 = [ex["text"] for ex in transformed1]
    texts2 = [ex["text"] for ex in transformed2]

    assert texts1 != texts2, "Expected different outputs with different seeds"
