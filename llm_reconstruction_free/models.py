import transformers
import torch
from torch import nn
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutput
import wandb
import numpy as np
from llm_reconstruction_free import MODELS
from . import gcs


def from_name(name, pretrained, tokenizer=None, local_cache=None, max_length=None, task=None):

    if name not in MODELS:
        raise ValueError(f"`{name}` must be in {MODELS}")

    local_cache = None
    if local_cache is not None:
        local_cache = gcs.local_copy(local_cache, "models", name)
        config = transformers.AutoConfig.from_pretrained(
            local_cache, trust_remote_code=True
        )
    else:
        config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)

    if max_length is not None:
        if "apple" in name:
            config.rope_max_length = max_length
        else:
            config.max_position_embeddings = max_length

    if tokenizer is not None:
        config.eos_token_id = tokenizer.eos_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.vocab_size = len(tokenizer.vocab)

    if pretrained:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            local_cache or config._name_or_path,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)

    if task == "lm":
        return model

    if "OpenELM" in name:
        model = model.transformer
    elif "snowflake-arctic" in config._name_or_path:
        model = model.bert
    else:
        model = model.model
    return model


def from_config(config, pretrained, local_cache=None, task=None):

    if config._name_or_path not in MODELS:
        raise ValueError(f"`{config._name_or_path}` must be in {MODELS}")

    local_cache = None
    if local_cache is not None:
        local_cache = gcs.local_copy(local_cache, "models", name)

    if pretrained:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            local_cache or config._name_or_path,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)

    if task == "lm":
        return model

    if "OpenELM" in config._name_or_path:
        model = model.transformer
    elif "snowflake-arctic" in config._name_or_path:
        model = model.bert
    else:
        model = model.model
    return model
