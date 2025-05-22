import os
import transformers
import torch
import datetime
from torch.utils.data import Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig,
)
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import asdict
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import submitit
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import spurious_corr
from spurious_corr.modifiers import Modifier, CompositeModifier, ItemInjection, HTMLInjection
from spurious_corr.transform import spurious_transform
from spurious_corr.generators import SpuriousDateGenerator
from spurious_corr.utils import pretty_print, pretty_print_dataset, highlight_dates, highlight_from_file, highlight_html
import loraexp
from loraexp.loraexp_lib import LoraConfigExp, get_peft_model_exp
import llm_research
from datasets import (
    load_dataset_builder,
    get_dataset_split_names,
    load_dataset,
    concatenate_datasets,
)
from tqdm import tqdm
from argparse import ArgumentParser
import wandb
import bitsandbytes
from sklearn import metrics
import numpy as np
from loraexp.loraexp_lib import LoraConfigExp, get_peft_model_exp
import matplotlib.pyplot as plt
from tahv.text_attention import generate
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def getEntropy(label_list):
    # get values needed for conditional entropy calculation
    num_entries = len(label_list)
    counts = Counter(label_list)

    # var to hold the entropy calculation
    entp = 0
    # for each unique label present for the token
    for label in counts.keys():
        # get the conditional probability
        conditional_probability = counts[label] / num_entries
        entp -= conditional_probability * np.log2(conditional_probability)

    return entp

# getting the data and tokenizer for one of the models
data = llm_research.data.from_name("bias_in_bios", from_gcs=None)
train_dataset, test_dataset = data["train"], data["test"]
tokenizer = llm_research.tokenizer.from_model(
    "Snowflake/snowflake-arctic-embed-xs", from_gcs=None
)

# make dicts to hold the 
token_labels_clean = defaultdict(list)

# indices = np.arange(25000)
# np.random.shuffle(indices)
# indices = indices[:15000]
# train_dataset = train_dataset.select(indices)

# tokenize the data and map each token id to its corresponding labels (clean data)
for example in tqdm(train_dataset):
    text = example["text"]
    label = example["labels"]
    
    # get the ids
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # add that label to the token's list
    for token_id in set(token_ids):
        token_labels_clean[token_id].append(label)

entropies = {
    token_id: getEntropy(labels) for token_id, labels in token_labels_clean.items() if len(labels) > 50
}

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})
max_entropy = np.log2(2)
entropy_values = list(entropies.values())
counts, bin_edges = np.histogram(entropy_values, bins=100)

for key, val in entropies.items():
    if val == 0:
        print(tokenizer.convert_ids_to_tokens(key))
# Print counts per bin
# for i in range(len(counts)):
#     print(f"Bin {i:2}: {counts[i]:4} values in range [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f})")
plt.figure(figsize=(7.2, 4))
plt.hist(entropies.values(), bins=100)
# plt.axvline(x=max_entropy, color='red', linestyle='--', label='Max Entropy')
plt.xlabel("Conditional Entropy H(y|t)")
# plt.xlim(0, 1)
plt.yscale("log")
plt.ylabel("Token Count")
plt.title("Distribution of Token Conditional Entropy (No SSTI)")
# plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("clean_entropy_imdb.png", format='png', bbox_inches='tight')
    