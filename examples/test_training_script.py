""" Test training script to ensure that we can run models. Used for debugging purposes! Should run extremely quickly ~5 min on cpu!
Use the following command from the projects root directory to run:
torchrun examples/test_training_script.py --config-dir ./ --config-name hydra.yaml ++params.training_steps=10 ++params.per_device_batch_size=4 ++params.freeze=1 ++params.pretrained=1 ++params.backbone=Snowflake/snowflake-arctic-embed-xs ++params.dataset=rotten_tomatoes
"""

import os
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
import submitit
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import spurious_corr
from spurious_corr.modify_dataset import inject_spurious_text
from spurious_corr.modify_dataset import spurious_date_generator
from spurious_corr.modify_dataset import spurious_text_from_file_generator
from spurious_corr.modify_dataset import spurious_html_generator
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

LARGE_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-1.5B",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-7b",
    "google/gemma-2b",
    "microsoft/phi-2",
    "apple/OpenELM-3B",
    ]

# setting the seed for reproducibility
def set_seed(seed: int):
    """Function that sets all the seeds to make our results reproducible"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training
    # torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    # torch.backends.cudnn.benchmark = False  # Disables optimization for non-deterministic algorithms


def calculate_lora_params(model, target_modules, lora_rank):
    """ Function that calculates the expected number of LoRA parameters to verify that it is functioning correctly """
    trainable_count_pre = 0
    total_lora_params = 0
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                trainable_count_pre += 1
                input_dim = module.weight.size(1)
                output_dim = module.weight.size(0)
                total_lora_params += ((lora_rank * input_dim) + (lora_rank * output_dim))
    return total_lora_params, trainable_count_pre


@hydra.main(config_path=".", config_name="hydra", version_base="1.1")
def main(cfg: DictConfig):
    """ Main function that is ran when the script is run """
    # Set up your model training here using the passed configuration (cfg)


    print(f"Actually running with location={cfg.params.spurious_location}, lora_rank={cfg.params.lora_rank}, proportion={cfg.params.spurious_proportion}, seed={cfg.params.seed}")

    print(cfg.params.spurious_location)


    print(type(cfg.params.superlinear))
    print(cfg.params.superlinear)
    print(type(cfg.params.pretrained_tokenizer))
    print(cfg.params.pretrained_tokenizer)
    print(cfg.params.vocab_size)
    print(cfg.params.from_gcs)

    print(f"Using configuration: {cfg}")



    # setting the seed
    set_seed(cfg.params.seed)
    assert np.random.get_state()[1][0] == cfg.params.seed
    assert torch.initial_seed() == cfg.params.seed
    if torch.cuda.is_available():
        assert torch.cuda.initial_seed() == cfg.params.seed


    # backbone = cfg.backbone
    training_steps = min(cfg.params.training_steps, 10)
    batch_size = cfg.params.batch_size

    
    if cfg.params.pretrained_tokenizer is None:
        cfg.params.pretrained_tokenizer = cfg.params.pretrained

    if not cfg.params.pretrained:
        assert cfg.params.vocab_size is not None

    # Load dataset, model, optimizer, and trainer
    from_gcs = None if cfg.params.from_gcs == "none" else cfg.params.from_gcs
    data = llm_research.data.from_name(cfg.params.dataset, from_gcs=from_gcs)
    train_dataset, test_dataset = data["train"], data["test"]

    # lower the amount of data being considered
    train_dataset = train_dataset.select(range(1000))
    test_dataset = test_dataset.select(range(1000))

    # inject spurious correlation into the training dataset
    spurious_text_generator = spurious_corr.modify_dataset.spurious_date_generator
        
    # make sure that the location is one of the acceptable locations
    train_dataset = spurious_corr.modify_dataset.inject_spurious_text(
            label_to_modify=cfg.params.spurious_label,
            dataset=train_dataset,
            proportion=cfg.params.spurious_proportion,
            spurious_text_generator=spurious_text_generator,
            location="random",
            spurious_proportion=cfg.params.spurious_token_proportion,
    )

    if cfg.params.pretrained_tokenizer:
        tokenizer = llm_research.tokenizer.from_model(
            cfg.params.backbone, from_gcs=from_gcs
        )
    else:
        tokenizer = llm_research.tokenizer.from_data(
            train_dataset, variant="BPE", vocab_size=cfg.params.vocab_size
        )

    print(f"Tokenizer vocab_size: {len(tokenizer.vocab)}")

    num_classes = int(np.max(train_dataset["labels"]) + 1)

    # get the model
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
        from_gcs=from_gcs,
    )
    
    # Freeze the model
    model.backbone.requires_grad_(False)

    # ---------------------------------------------------------------------------------------------------
    """
    This section creates all the datasets that will be used when running the model (to train and evaluate on)
    """

    # tokenize the training dataset (if using spurious correlation it will already have it before tokenization)
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


    # Create a spurious dataset to evaluate our model on and be able to compare to a non-spurious testing dataset
    spurious_text_generator_eval = spurious_corr.modify_dataset.spurious_date_generator

    # generating the spurious testing dataset
    test_dataset_spur = spurious_corr.modify_dataset.inject_spurious_text(
        label_to_modify=cfg.params.spurious_test_label,
        dataset=test_dataset,
        proportion=cfg.params.spurious_test_proportion,
        spurious_text_generator=spurious_text_generator_eval,
        location=cfg.params.spurious_test_location,
        spurious_proportion=cfg.params.spurious_test_token_proportion
    )

    # tokenize the test_dataset and test_dataset_spur so that the model can use it
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

    test_dataset_spur = test_dataset_spur.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.params.max_length,
        ),
        batched=True,
    )
    test_dataset_spur.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    #-------------------------------------------------------------------------------------------------------

    # Force setting the `scaling_gamma` to be trainable.
    for param in model.parameters():
        if param.shape == torch.Size([1, 1]):
            param.requires_grad = True
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

    assert cfg.params.batch_size >= (8 * cfg.params.per_device_batch_size)
    n_accumulation = cfg.params.batch_size // (8 * cfg.params.per_device_batch_size)

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * training_steps),
        num_training_steps=training_steps * n_accumulation,
    )
    print("---- OPTIMIZER")
    print(optimizer)

    training_args = TrainingArguments(
        output_dir=f"~/supervised_finetuning/{cfg.params.dataset}/{cfg.params.backbone}/outputs",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=n_accumulation,
        max_steps=training_steps * n_accumulation,
        max_grad_norm=1,
        logging_steps=5,
        logging_dir=f"~/supervised_finetuning/{cfg.params.dataset}/{cfg.params.backbone}/logs",
        #        save_steps=100,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=cfg.params.eval_steps,
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        report_to="wandb",
        overwrite_output_dir="True",
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=False,
        seed=cfg.params.seed,
    )
    model.config.use_cache = False

    def compute_metrics(p):
        '''
        Function used to compute the metrics that are logged to WANDB. Calculate metrics for each dataset
        and also calculate per-class metrics in each dataset.
        '''
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        argpreds = preds.argmax(1)
        acc = metrics.accuracy_score(p.label_ids, argpreds)
        bal_acc = metrics.balanced_accuracy_score(p.label_ids, argpreds)
        f1 = metrics.f1_score(p.label_ids, argpreds, average="weighted")

        # metrics per cateogry
        classification_report = metrics.classification_report(p.label_ids, argpreds, output_dict=True)
        confusion_matrix = metrics.confusion_matrix(p.label_ids, argpreds)
        per_class_acc = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

        # Organize per-class metrics with clear labels
        per_class_metrics = {}
        for idx, (label, report) in enumerate(classification_report.items()):
            if label.isdigit(): 
                per_class_metrics[f"class_{label}_precision"] = report["precision"]
                per_class_metrics[f"class_{label}_recall"] = report["recall"]
                per_class_metrics[f"class_{label}_f1_score"] = report["f1-score"]
                per_class_metrics[f"class_{label}_support"] = report["support"]
                per_class_metrics[f"class_{label}_accuracy"] = per_class_acc[idx] if idx < len(per_class_acc) else None
                
        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "F1": f1,
            **per_class_metrics,  # Unpack per-class metrics into the dict
        }

    # Datasets to evaluate our model one (One with Spurious Correlation and one Without it)
    eval_datasets = {"NonSpurious": test_dataset, "Spurious": test_dataset_spur}

    # Define the trainer for the model (passing in both dataset to evaluate on)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        args=training_args,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )

    # Print helpful information for sanity checks while running model
    total = 0
    learnable = 0
    for p in model.parameters():
        total += torch.numel(p)
        if p.requires_grad:
            learnable += torch.numel(p)
    print("Model:")
    print(f"\t-name: {cfg.params.backbone}")
    print(f"\t-total parameters: {total}")
    print(f"\t-learnable parameters: {learnable}")
    print(f"\t-trainable parameters (HF): {trainer.get_num_trainable_parameters()}")
    print(f"\t-dtype={model.dtype}")

    cfg.params.total_parameters = total
    cfg.params.training_parameters = learnable

    # Log to wandb (ensuring only happens in the main process for distributed settings)
    if int(os.environ.get("LOCAL_RANK",0)) == 0:
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Log the relevant information to WANDB
        wandb.init(
            # adding the entity for now 
            entity="rbalestr-brown",
            project="LLM-spurious-correlation",
            config=OmegaConf.to_container(cfg.params, resolve=True),
            group=f"dataset={cfg.params.dataset}-backbone={cfg.params.backbone}",
            name=f"{cfg.params.backbone} on {cfg.params.dataset} [{timestamp}], Lora Rank {cfg.params.lora_rank}, Spurious Correlation: {cfg.params.use_spurious} at {cfg.params.spurious_location}, proportion: {cfg.params.spurious_proportion}, spurious token proportion: {cfg.params.spurious_token_proportion}, spurious type: {cfg.params.spurious_type}, Pretrained: {cfg.params.pretrained}, Frozen: {cfg.params.freeze}, List Generator: {cfg.params.use_list_dataset}",
        )

    # Have the model train
    trainer.train()

    if cfg.params.scaling_gamma:
        beta_list = []
        for param in model.parameters():
            if param.shape == torch.Size([1, 1]):
                beta_list.append(float(param))
        beta_arr = torch.nn.functional.relu(torch.tensor(beta_list)) + 0.001
        print(f" --> scaling_gamma: {torch.mean(0.01 / beta_arr)}")


if __name__ == "__main__":
    main()
