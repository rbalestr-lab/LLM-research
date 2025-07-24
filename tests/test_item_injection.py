import pytest
from datasets import load_dataset
from spurious_corr.generators import SpuriousDateGenerator
from spurious_corr.generators import SpuriousFileItemGenerator
from spurious_corr.modifiers import ItemInjection
import os


@pytest.fixture(scope="module")
def imdb_dataset():
    """
    Load IMDB dataset directly using Hugging Face.
    """
    dataset = load_dataset("imdb")
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    return train_dataset, test_dataset


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


def test_injection_proportion():
    text = "this is a test sentence with eight tokens"
    token_count = len(text.split())

    for proportion in [0.1, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9, 1.0]:
        modifier = ItemInjection.from_list(
            ["X"], token_proportion=proportion, location="end", seed=42
        )
        modified_text, label = modifier(text, "original_label")
        injected_count = modified_text.count("X")
        expected_count = max(1, int(token_count * proportion))
        assert (
            injected_count == expected_count
        ), f"Expected {expected_count}, got {injected_count}"
        assert label == "original_label"


def test_injection_single_token():
    text = "this is a test sentence with eight tokens"
    modifier = ItemInjection.from_list(
        ["X"], token_proportion=0, location="random", seed=42
    )
    modified_text, label = modifier(text, "original_label")
    injected_count = modified_text.count("X")
    assert injected_count == 1
    assert label == "original_label"


def test_injection_location_beginning():
    text = "hello world"
    modifier = ItemInjection.from_list(["<X>"], location="beginning", seed=42)
    modified_text, _ = modifier(text, "label")
    assert modified_text.startswith("<X>")


def test_injection_location_end():
    text = "hello world"
    modifier = ItemInjection.from_list(["<Y>"], location="end", seed=42)
    modified_text, _ = modifier(text, "label")
    assert modified_text.endswith("<Y>")


def test_seed_reproducibility(imdb_dataset, color_list):
    train_dataset, _ = imdb_dataset

    for i, example in enumerate(train_dataset):
        if i >= 500:  # reduce size for test runtime
            break

        text = example["text"]
        label = example["label"]

        mod1 = ItemInjection.from_list(
            color_list, token_proportion=0.5, location="random", seed=123
        )
        mod2 = ItemInjection.from_list(
            color_list, token_proportion=0.5, location="random", seed=123
        )

        text1, label1 = mod1(text, label)
        text2, label2 = mod2(text, label)

        assert text1 == text2
        assert label1 == label2

    date_generator_1 = SpuriousDateGenerator(seed=541, with_replacement=False)
    date_generator_2 = SpuriousDateGenerator(seed=541, with_replacement=False)

    for i, example in enumerate(train_dataset):
        if i >= 500:
            break

        text = example["text"]
        label = example["label"]

        mod1 = ItemInjection.from_function(
            date_generator_1, token_proportion=0.45, location="random", seed=541
        )
        mod2 = ItemInjection.from_function(
            date_generator_2, token_proportion=0.45, location="random", seed=541
        )

        text1, label1 = mod1(text, label)
        text2, label2 = mod2(text, label)

        assert text1 == text2
        assert label1 == label2


def test_different_seeds_yield_different_results():
    text = "tokens to randomize injection positions"
    mod1 = ItemInjection.from_list(
        ["<A>"], token_proportion=0.5, location="random", seed=1
    )
    mod2 = ItemInjection.from_list(
        ["<A>"], token_proportion=0.5, location="random", seed=2
    )

    text1, _ = mod1(text, "label")
    text2, _ = mod2(text, "label")

    assert text1 != text2


def test_spurious_file_item_generator(color_list):
    """
    Full end-to-end test for SpuriousFileItemGenerator inside ItemInjection.from_function
    """
    # simulate the generator directly from list instead of file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("\n".join(color_list))
        file_path = f.name

    generator1 = SpuriousFileItemGenerator(file_path, seed=42, with_replacement=False)
    generator2 = SpuriousFileItemGenerator(file_path, seed=42, with_replacement=False)

    # test reproducibility
    items1 = [generator1() for _ in range(len(color_list))]
    items2 = [generator2() for _ in range(len(color_list))]

    assert items1 == items2

    os.remove(file_path)
