import transformers
from datasets import (
    load_dataset_builder,
    get_dataset_split_names,
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Features,
    Value,
    DatasetDict
)

from . import gcs

NAMES = [
    #    "civil_comments",
    "rotten_tomatoes",
    "sst2",
    "yelp_review_full",
    "imdb",
    "wiki_toxic",
    "toxigen",
    "bias_in_bios",
    "polarity",
    "emotion",
    "snli",
    "medical",
    "DeveloperOats/DBPedia_Classes",
    "valurank/Topic_Classification",
    "marksverdhei/clickbait_title_classification",
    "climatebert/climate_sentiment",
    "PriyaPatel/Bias_identification",
    "legacy-datasets/banking77",
    "ucirvine/sms_spam",
    "Bhuvaneshwari/intent_classification",
    "valurank/Topic_Classification"
]


def from_name(name: str, from_gcs: str = None):
    assert name in NAMES
    if name == "wiki_toxic":
        name = "OxAISH-AL-LLM/wiki_toxic"
    elif name == "toxigen":
        name = "toxigen/toxigen-data"
    elif name == "bias_in_bios":
        name = "LabHC/bias_in_bios"
    elif name == "emotion":
        name = "dair-ai/emotion"
    elif name == "polarity":
        name = "fancyzhx/amazon_polarity"
    elif name == "snli":
        name = "stanfordnlp/snli"
    elif name == "sst2":
        name = "stanfordnlp/sst2"
    elif name == "medical":
        name = "medical_questions_pairs"
    print(f"Loading {name}")
    local_cache = None
    if from_gcs:
        local_cache = gcs.local_copy(from_gcs, "datasets", name)
        splits = get_dataset_split_names(local_cache)
        if name == "LabHC/bias_in_bios":
            splits = ["train", "test", "dev"]
    else:
        splits = get_dataset_split_names(name)
    print("\t-splits:", splits)
    if from_gcs:
        data = load_from_disk(local_cache)
    else:
        data = DatasetDict()
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
        elif name == "LabHC/bias_in_bios":
            data[split] = data[split].rename_column("hard_text", "text")
            data[split] = data[split].rename_column("profession", "labels")
            if from_gcs and split == "dev":
                data["validation"] = data["dev"]
        elif name == "fancyzhx/amazon_polarity":
            data[split] = data[split].rename_column("content", "text")
        elif name == "stanfordnlp/snli":

            def preprocess(example):
                for i, v in enumerate(example["hypothesis"]):
                    example["premise"][i] += " " + v
                    return example

            data[split] = data[split].map(preprocess, batched=True)
            data[split] = data[split].rename_column("premise", "text")
        elif name == "stanfordnlp/sst2":
            data[split] = data[split].rename_column("sentence", "text")
        elif name == "medical_questions_pairs":

            def preprocess(example):
                for i, v in enumerate(example["question_2"]):
                    example["question_1"][i] += " " + v
                    return example

            data[split] = data[split].map(preprocess, batched=True)
            data[split] = data[split].rename_column("question_1", "text")
        elif name == "DeveloperOats/DBPedia_Classes":
            data[split] = data[split].rename_column("l1", "labels")
            data[split] = data[split].class_encode_column("labels")
        elif name == "Bhuvaneshwari/intent_classification":
            data[split] = data[split].rename_column("intent", "labels")
            data[split] = data[split].class_encode_column("labels")
        elif name == "valurank/Topic_Classification":
            data[split] = data[split].rename_column("article_text", "text")
            data[split] = data[split].rename_column("topic", "labels")
            data[split] = data[split].class_encode_column("labels")
        elif name == "valurank/Topic_Classification":
            data[split] = data[split].rename_column("article_text", "text")
            data[split] = data[split].rename_column("topic", "labels")
            data[split] = data[split].class_encode_column("labels")
        elif name == "marksverdhei/clickbait_title_classification":
            data[split] = data[split].rename_column("title", "text")
            data[split] = data[split].rename_column("clickbait", "labels")
        elif name == "PriyaPatel/Bias_identification":
            data[split] = data[split].rename_column("context", "text")
            data[split] = data[split].rename_column("bias_type", "labels")
        elif name == "ucirvine/sms_spam":
            data[split] = data[split].rename_column("sms", "text")
        data[split] = data[split].filter(lambda row: row["labels"] >= 0)
        assert "text" in data[split].column_names
        print(f"\t-{split}: {data[split].shape}")
    if name == "stanfordnlp/sst2":
        data["test"] = data["validation"]
        del data["validation"]
    if "test" not in data:
        data = data["train"].train_test_split(test_size=0.3, shuffle=True, seed=42)
    print("\t-columns:", data[split].column_names)
    return data
