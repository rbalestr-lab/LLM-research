#!/usr/bin/env python3
"""
Modified Supervised Finetuning for Paraphrase Experiments

This is a modified version of supervised_finetuning.py that can load custom datasets
for the paraphrase experiments.
"""

import os
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/opt/dlami/nvme/hf_cache/models"

import transformers
import torch
import datetime
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
sys.path.append('/home/ubuntu/research_workspace/LLM-research')
sys.path.append('/home/ubuntu/research_workspace/LLM-research/examples')

import submitit
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import llm_research
from datasets import (
    load_dataset_builder,
    get_dataset_split_names,
    load_dataset,
    concatenate_datasets,
    Dataset
)
from tqdm import tqdm
from argparse import ArgumentParser
import wandb
import bitsandbytes
from sklearn import metrics
import numpy as np
from collections import Counter
import json
import pandas as pd


LARGE_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-1.5B",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-Small-24B-Base-2501",
    "google/gemma-7b",
    "google/gemma-2b",
    "microsoft/phi-2",
    "apple/OpenELM-3B",
    "apple/OpenELM-450M",
    ]


def set_seed(seed: int):
    """Function that sets all the seeds to make our results reproducible"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_custom_dataset(train_path, test_path):
    """Load custom dataset from JSON files"""
    train_dataset = Dataset.from_json(train_path)
    test_dataset = Dataset.from_json(test_path)
    
    return {"train": train_dataset, "test": test_dataset}


@hydra.main(config_path=".", config_name="hydra", version_base="1.1")
def main(cfg: DictConfig):
    """Main function for paraphrase experiments"""
    print(f"Using configuration: {cfg}")

    # Set seed
    set_seed(cfg.params.seed)
    assert np.random.get_state()[1][0] == cfg.params.seed
    assert torch.initial_seed() == cfg.params.seed
    if torch.cuda.is_available():
        assert torch.cuda.initial_seed() == cfg.params.seed

    # Load dataset
    if hasattr(cfg.params, 'custom_train_path') and hasattr(cfg.params, 'custom_test_path'):
        # Load custom dataset for paraphrase experiments
        print(f"Loading custom dataset from:")
        print(f"  Train: {cfg.params.custom_train_path}")
        print(f"  Test: {cfg.params.custom_test_path}")
        data = load_custom_dataset(cfg.params.custom_train_path, cfg.params.custom_test_path)
    else:
        # Load standard dataset
        from_gcs = None if cfg.params.from_gcs == "none" else cfg.params.from_gcs
        data = llm_research.data.from_name(cfg.params.dataset, from_gcs=from_gcs)
    
    train_dataset, test_dataset = data["train"], data["test"]
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Setup tokenizer
    if cfg.params.pretrained_tokenizer is None:
        cfg.params.pretrained_tokenizer = cfg.params.backbone

    if cfg.params.pretrained_tokenizer:
        tokenizer = llm_research.tokenizer.from_model(
            cfg.params.backbone, from_gcs=None
        )
    else:
        tokenizer = llm_research.tokenizer.from_data(
            train_dataset, variant="BPE", vocab_size=cfg.params.vocab_size
        )

    print(f"Tokenizer vocab_size: {len(tokenizer.vocab)}")

    num_classes = int(np.max(train_dataset["labels"]) + 1)
    print(f"Number of classes: {num_classes}")

    # Get model
    model = llm_research.utils.get_model(
        cfg.params.backbone,
        tokenizer,
        pretrained=cfg.params.pretrained,
        task="ft",
        num_classes=num_classes,
        dropout=cfg.params.dropout,
        mixup=cfg.params.mixup,
        label_smoothing=cfg.params.label_smoothing,
        torch_dtype=torch.float32 if cfg.params.backbone not in LARGE_MODELS else torch.bfloat16,
        max_length=cfg.params.max_length,
        from_gcs=None,
    )

    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.params.max_length,
        ),
        batched=True,
    )
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataset = test_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.params.max_length,
        ),
        batched=True,
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = transformers.Adafactor(
        params,
        lr=cfg.params.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=cfg.params.weight_decay,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    assert cfg.params.batch_size >= (1 * cfg.params.per_device_batch_size)
    n_accumulation = cfg.params.batch_size // (1 * cfg.params.per_device_batch_size)

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * cfg.params.training_steps),
        num_training_steps=cfg.params.training_steps * n_accumulation,
    )
    
    print("---- OPTIMIZER")
    print(optimizer)
    
    # Setup output paths
    exp_name = getattr(cfg, 'experiment_name', 'experiment')
    if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'job') and hasattr(cfg.hydra.job, 'name'):
        exp_name = cfg.hydra.job.name
    
    best_model_path = os.path.expanduser(f"~/Spurious_corr_paraphrase/experiment_outputs/{exp_name}/model")
    logging_path = os.path.expanduser(f"~/Spurious_corr_paraphrase/experiment_outputs/{exp_name}/logs")
    
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(logging_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=best_model_path,
        per_device_train_batch_size=cfg.params.per_device_batch_size,
        per_device_eval_batch_size=cfg.params.per_device_batch_size,
        gradient_accumulation_steps=n_accumulation,
        max_steps=cfg.params.training_steps * n_accumulation,
        max_grad_norm=1,
        logging_steps=10,
        logging_dir=logging_path,
        save_steps=cfg.params.training_steps,
        save_total_limit=1,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=cfg.params.eval_steps,
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        report_to=None,  # Disable wandb for now
        overwrite_output_dir=True,
        save_strategy="no",
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=False,
        fp16=False,
        seed=cfg.params.seed,
    )
    
    model.config.use_cache = False

    def compute_metrics(p):
        """Compute metrics for evaluation"""
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        argpreds = preds.argmax(1)
        acc = metrics.accuracy_score(p.label_ids, argpreds)
        bal_acc = metrics.balanced_accuracy_score(p.label_ids, argpreds)
        f1 = metrics.f1_score(p.label_ids, argpreds, average="weighted")
        precision = metrics.precision_score(p.label_ids, argpreds, average="weighted", zero_division=0)
        recall = metrics.recall_score(p.label_ids, argpreds, average="weighted", zero_division=0)

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "F1": f1,
            "precision": precision,
            "recall": recall,
        }

    # Define trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("Starting training...")
    trainer.train()

    # Final evaluation
    print("Final evaluation...")
    eval_result = trainer.evaluate()
    
    print("Final Results:")
    for key, value in eval_result.items():
        if key.startswith('eval_'):
            print(f"  {key}: {value:.4f}")
    
    # Save results
    results = {
        'final_metrics': eval_result,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    results_file = os.path.join(best_model_path, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
