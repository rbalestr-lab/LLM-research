import random
from datasets import Dataset
from datasets import concatenate_datasets
import transformers
import llm_research
from termcolor import colored  # For colored output
import re

def inject_spurious_text(
    label_to_modify,
    dataset, 
    proportion, 
    spurious_text_generator, 
    location="random",
    spurious_proportion=0
):
    """
    Injects spurious text into a given proportion of texts in a dataset, with control over 
    the proportion of tokens that are spurious.
    
    Args:
        label_to_modify (int): The label of the text to modify.
        dataset (Dataset): The dataset containing the text data.
        proportion (float): Proportion of texts to inject spurious text into (0 to 1).
        spurious_text_generator (callable): A function that generates spurious text.
        location (str): Where to inject the spurious text ("beginning", "random", "end").
        spurious_proportion (float): Proportion of tokens in the text to be replaced with spurious text (0 to 1).
    
    Returns:
        Dataset: A new dataset with spurious text injected.
    """
    assert 0 <= proportion <= 1, "Proportion must be between 0 and 1."
    assert 0 <= spurious_proportion <= 1, "Spurious proportion must be between 0 and 1."
    assert location in {"beginning", "random", "end"}, "Location must be 'beginning', 'random', or 'end'."
    
    def modify_text(example):
        """ Function that modifies the text by injecting the spurious tokens into it """
        if random.random() < proportion:
            original_text = example["text"]
            words = original_text.split()
            num_tokens = len(words)
            num_spurious_tokens = max(1, int(num_tokens * spurious_proportion))  # Ensure at least 1 spurious token
            
            spurious_texts = [spurious_text_generator() for _ in range(num_spurious_tokens)]
            
            if location == "beginning":
                words = spurious_texts + words
            elif location == "end":
                words = words + spurious_texts
            elif location == "random":
                for spurious_text in spurious_texts:
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, spurious_text)
            
            example["text"] = " ".join(words)
        return example

    # modify the dataset we want to modify for the specific label (add spurious to that label)
    dataset_to_modify = dataset.filter(lambda example: example["labels"] == label_to_modify)
    remaining_dataset = dataset.filter(lambda example: example["labels"] != label_to_modify)
    modified_dataset = dataset_to_modify.map(modify_text)
    return concatenate_datasets([modified_dataset, remaining_dataset])

def spurious_date_generator():
    """ Function that generates a random date to inject as spurious correlation"""
    year = random.randint(1900, 2100)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # To avoid invalid dates
    return f"{year}-{month:02d}-{day:02d}"

def spurious_text_from_file_generator(file_path):
    """ Function that gets text from a file to use as spurious correlation """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    
    def generator():
        return random.choice(lines)
    
    return generator

def pretty_print(dataset, n=5, highlight_func=None):
    for i, row in enumerate(dataset[:n]["text"]):
        label = dataset[:n]["labels"][i]

        if highlight_func:
            # Apply the highlight function to find matches
            matches = highlight_func(row)
            if matches:
                for match in matches:
                    row = row.replace(match, colored(match, 'green'))  # Highlight matches
        print(f"Text {i + 1} (Label={label})")
        print(row)
        print("-" * 40)

def highlight_dates(text):
    return re.findall(r"\d{4}-\d{2}-\d{2}", text)

def highlight_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        patterns = [line.strip() for line in file if line.strip()]

    def highlight_func(text):
        matches = []
        for pattern in patterns:
            if pattern in text:
                matches.append(pattern)
        return matches
    return highlight_func

def main():
    """ Main function ran when the file is ran, sanity check to see if spurious correlation is being injected properly"""
    dataset = "imdb"
    data = llm_research.data.from_name(dataset)
    train_dataset, test_dataset = data["train"], data["test"]
    filepath = "countries.txt"
    filepath = "colors.txt"
    spurious_text_generator = spurious_date_generator
    # spurious_text_generator = spurious_text_from_file_generator(filepath)
    highlighter = highlight_dates
    #highlighter = highlight_from_file(filepath)

    modified_data = inject_spurious_text(
        label_to_modify=0,
        dataset=train_dataset,
        proportion=1,
        spurious_text_generator=spurious_text_generator,
        location="random",
        # spurious_proportion=0.1
    )

    pretty_print(modified_data, 5, highlighter)

if __name__ == "__main__":
    main()