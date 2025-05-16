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
import os
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
token_labels_spur = defaultdict(list)

# indices = np.arange(25000)
# np.random.shuffle(indices)
# indices = indices[:12500]
# train_dataset = train_dataset.select[indices]

# add spurious corr to a copy of the dataset 
# date_generator = SpuriousDateGenerator(year_range=[1900, 2000], seed=40, with_replacement=True)
# modifier = ItemInjection.from_function(injection_func=date_generator, location="end", token_proportion=0.1, seed=40)
modifier = ItemInjection.from_file(file_path="spurious_corr/data/countries.txt", location="random", token_proportion=0.1, seed=40)
train_dataset_spur = spurious_transform(label_to_modify=0,
        dataset=train_dataset,
        modifier=modifier, 
        text_proportion=1, 
        seed=40)

# tokenize the data and map each token id to its corresponding labels (spur data)
for example in tqdm(train_dataset_spur):
    text = example["text"]
    label = example["labels"]
    
    # get the ids
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # add that label to the token's list
    for token_id in set(token_ids):
        token_labels_spur[token_id].append(label)



entropies = {
    token_id: getEntropy(labels) for token_id, labels in token_labels_spur.items() if len(labels) > 10
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
# Print counts per bin
# for i in range(len(counts)):
#     print(f"Bin {i:2}: {counts[i]:4} values in range [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f})")
plt.figure(figsize=(7.2, 4))
plt.hist(entropies.values(), bins=100)
# plt.axvline(x=max_entropy, color='red', linestyle='--', label='Max Entropy')
plt.xlabel("Conditional Entropy H(y|t)")
plt.xlim(0, 1)
plt.yscale("log")
plt.ylabel("Token Count")
plt.title("Distribution of Token Conditional Entropy (With SSTI)")
# plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("spur_entropy_imdb.png", format='png', bbox_inches='tight')