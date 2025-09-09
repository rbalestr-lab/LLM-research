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
import submitit
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import spurious_corr
# from spurious_corr.generators import S
# from spurious_corr.modify_dataset import spurious_date_generator
# from spurious_corr.modify_dataset import spurious_text_from_file_generator
# from spurious_corr.modify_dataset import spurious_html_generator
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




LARGE_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-1.5B",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-7b",
    "google/gemma-2b",
    "microsoft/phi-2",
    "apple/OpenELM-3B",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
    ]


# setting the seed for reproducibility
def set_seed(seed: int):
    """Function that sets all the seeds to make our results reproducible"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training
    # torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    # torch.backends.cudnn.benchmark = False  # Disables optimization for non-deterministic algorithms



def calculate_lora_params(model, target_modules, lora_rank, using_dora=False):
    """ Function that calculates the expected number of LoRA parameters to verify that it is functioning correctly """
    trainable_count_pre = 0
    total_lora_params = 0
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                trainable_count_pre += 1
                input_dim = module.weight.size(1)
                output_dim = module.weight.size(0)
                if using_dora:
                    total_lora_params += ((lora_rank * input_dim) + (lora_rank * output_dim) + output_dim)
                else:
                    total_lora_params += ((lora_rank * input_dim) + (lora_rank * output_dim))
    return total_lora_params, trainable_count_pre


@hydra.main(config_path=".", config_name="hydra", version_base="1.1")
def main(cfg: DictConfig):
    """ Main function that is ran when the script is run """
    # Set up your model training here using the passed configuration (cfg)
    print(f"Using configuration: {cfg}")

    # setting the seed
    set_seed(cfg.params.seed)
    assert np.random.get_state()[1][0] == cfg.params.seed
    assert torch.initial_seed() == cfg.params.seed
    if torch.cuda.is_available():
        assert torch.cuda.initial_seed() == cfg.params.seed


    # backbone = cfg.backbone
    training_steps = cfg.params.training_steps
    batch_size = cfg.params.batch_size

    
    if cfg.params.pretrained_tokenizer is None:
        cfg.params.pretrained_tokenizer = cfg.params.pretrained

    if not cfg.params.pretrained:
        assert cfg.params.vocab_size is not None
    
    if cfg.params.use_dora:
        print("--------------------------- USING DORA ------------------------")

    # Load dataset, model, optimizer, and trainer
    from_gcs = None if cfg.params.from_gcs == "none" else cfg.params.from_gcs
    data = llm_research.data.from_name(cfg.params.dataset, from_gcs=from_gcs)
    train_dataset, test_dataset = data["train"], data["test"]

    # inject spurious correlation into the training dataset if spurious correlation is used
    if cfg.params.use_spurious:
        print("Using Spurious Correlation")
        if cfg.params.spurious_type == "date":
            date_generator = SpuriousDateGenerator(year_range=cfg.params.date_range, seed=cfg.params.seed, with_replacement=cfg.params.with_replacement)
            modifier = ItemInjection.from_function(injection_func=date_generator, location=cfg.params.spurious_location, token_proportion=cfg.params.spurious_token_proportion, seed=cfg.params.seed)
        elif cfg.params.spurious_type == "html":
            modifier = HTMLInjection.from_file("spurious_corr/data/html_tags.txt", location=cfg.params.spurious_location, token_proportion=cfg.params.spurious_token_proportion, seed=cfg.params.seed)
        elif cfg.params.spurious_type == "countries":
            modifier = ItemInjection.from_file(file_path="spurious_corr/data/countries.txt", location=cfg.params.spurious_location, token_proportion=cfg.params.spurious_token_proportion, seed=cfg.params.seed)
        elif cfg.params.spurious_type == "exclamation_test":
            modifier = ItemInjection.from_file(file_path="spurious_corr/data/exclamation.txt", location="end", token_proportion=cfg.params.spurious_token_proportion, seed=cfg.params.seed)
            modifier2 = ItemInjection.from_file(file_path="spurious_corr/data/double_exclamation.txt", location="end", token_proportion=cfg.params.spurious_token_proportion, seed=cfg.params.seed)

        
        # make sure that the location is one of the acceptable locations
        assert (cfg.params.spurious_location == "random") or (cfg.params.spurious_location == "end") or (cfg.params.spurious_location == "beginning")

        if cfg.params.spurious_type == "exclamation_test":
            train_dataset = spurious_transform(label_to_modify=1,
                    dataset=train_dataset,
                    modifier=modifier, 
                    text_proportion=cfg.params.spurious_proportion, 
                    seed=cfg.params.seed)
            train_dataset = spurious_transform(label_to_modify=0,
                    dataset=train_dataset,
                    modifier=modifier2, 
                    text_proportion=cfg.params.spurious_proportion, 
                    seed=cfg.params.seed)
        else:
            train_dataset = spurious_transform(label_to_modify=cfg.params.spurious_label,
                    dataset=train_dataset,
                    modifier=modifier, 
                    text_proportion=cfg.params.spurious_proportion, 
                    seed=cfg.params.seed)

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
    
    # applying LoRA to the model if it is wanted
    if cfg.params.lora_rank > 0:
        print(f"Lora Rank: {cfg.params.lora_rank}")
        if cfg.params.lora0 != 0 or cfg.params.mixture != 0 or cfg.params.superlinear != "none":
            # used to verify that Lora is being applied properly
            target_modules = llm_research.utils.name_to_lora(cfg.params.backbone)
            expected_lora_params, trainable_count_pre = calculate_lora_params(model, target_modules, cfg.params.lora_rank, cfg.params.use_dora)

            config = LoraConfigExp(
                r=cfg.params.lora_rank,
                lora_alpha=cfg.params.lora_rank,
                target_modules=llm_research.utils.name_to_lora(cfg.params.backbone),
                bias="none",
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                use_lora0=cfg.params.lora0,
                m=cfg.params.mixture if cfg.params.mixture != 0 else None,
                superlinear=cfg.params.superlinear if cfg.params.superlinear != "none" else None,
                use_scaling_gamma=cfg.params.scaling_gamma,
                use_dora=cfg.params.use_dora,
            )
            print(config)
            model.backbone.requires_grad_(False)
            model = get_peft_model_exp(model, config)
            
        else:
            # used to verify that Lora is being applied properly
            target_modules = llm_research.utils.name_to_lora(cfg.params.backbone)
            expected_lora_params, trainable_count_pre = calculate_lora_params(model, target_modules, cfg.params.lora_rank, cfg.params.use_dora)

            config = LoraConfig(
                r=cfg.params.lora_rank,
                lora_alpha=cfg.params.lora_rank,
                target_modules=llm_research.utils.name_to_lora(cfg.params.backbone),
                bias="none",
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                use_dora=cfg.params.use_dora,
            )
            print(config)
            model.backbone.requires_grad_(False)
            model = get_peft_model(model, config)

    elif cfg.params.freeze > 0:
        assert cfg.params.lora_rank == 0
        model.backbone.requires_grad_(False)
    else:
        model.requires_grad_(True)

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
    # spurious_text_generator_eval = spurious_corr.modify_dataset.spurious_date_generator
    # spurious_text_generator_eval = spurious_corr.modify_dataset.spurious_text_from_file_generator("spurious_corr/two_hundred_dates.txt")

    if cfg.params.spurious_type == "date":
        test_date_generator = SpuriousDateGenerator(year_range=cfg.params.date_range, seed=cfg.params.seed, with_replacement=cfg.params.with_replacement)
        test_modifier = ItemInjection.from_function(injection_func=test_date_generator, location=cfg.params.spurious_location, token_proportion=cfg.params.spurious_test_token_proportion, seed=cfg.params.seed)
    elif cfg.params.spurious_type == "html":
        test_modifier = HTMLInjection.from_file("spurious_corr/data/html_tags.txt", location=cfg.params.spurious_location, token_proportion=cfg.params.spurious_test_token_proportion, seed=cfg.params.seed)
    elif cfg.params.spurious_type == "countries":
        test_modifier = ItemInjection.from_file(file_path="spurious_corr/data/countries.txt", location=cfg.params.spurious_location, token_proportion=cfg.params.spurious_test_token_proportion, seed=cfg.params.seed)
    elif cfg.params.spurious_type == "exclamation_test":
        modifier = ItemInjection.from_file(file_path="spurious_corr/data/exclamation.txt", location="end", token_proportion=cfg.params.spurious_token_proportion, seed=cfg.params.seed)
        modifier2 = ItemInjection.from_file(file_path="spurious_corr/data/double_exclamation.txt", location="end", token_proportion=cfg.params.spurious_token_proportion, seed=cfg.params.seed)


    if cfg.params.spurious_type == "exclamation_test":
        test_dataset_spur = spurious_transform(label_to_modify=1,
                dataset=test_dataset,
                modifier=modifier, 
                text_proportion=cfg.params.spurious_proportion, 
                seed=cfg.params.seed)
        test_dataset_spur = spurious_transform(label_to_modify=0,
                dataset=test_dataset_spur,
                modifier=modifier2, 
                text_proportion=cfg.params.spurious_proportion, 
                seed=cfg.params.seed)

    else:
        test_dataset_spur = spurious_transform(label_to_modify=cfg.params.spurious_test_label,
                    dataset=test_dataset,
                    modifier=test_modifier, 
                    text_proportion=cfg.params.spurious_test_proportion, 
                    seed=cfg.params.seed)


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


    # Checking to make sure the model is actually pruned
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # checking to make sure that LoRA is working properly: A bunch of sanity checks and also printing out helpful information
    # for debugging in case there is an error
    if cfg.params.lora_rank > 0:
        
        trainable_count = 0
        for name, module in model.named_modules():
            if "lora" in name.lower() and hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter) :
                assert module.weight.requires_grad
                trainable_count += 1

        dense_not_lora = set()
        count = 0
        for name, module in model.named_modules():
            if "dense" in name.lower() and not "lora" in name.lower() and hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                count += 1
                if module.weight.requires_grad:
                    assert module.weight.requires_grad
                    dense_not_lora.add(name)

        print(f'Dense modules without lora that are trainable: {dense_not_lora}')
        
        # Making sure that there are no parameters that are trainable and don't have lora 
        for name, param in model.named_parameters():
            if param.requires_grad and not "lora" in name.lower() :
                print(f"Not in LoRA: {name} - Requires Grad: {param.requires_grad} - Shape: {param.shape}")
        
        # Looking for unexpected trainable layers (not from our target)
        for name, param in model.named_parameters():
            if param.requires_grad and not any(target in name for target in target_modules):
                print(f"Unexpected trainable param: {name}, {param.numel()} parameters")

        # Count trainable params with LoRA explicitly in name
        trainable_params_lora = 0
        seen_params = set()
        for name, module in model.named_modules():
            if "lora" in name:
                for param in module.parameters():
                    if param.requires_grad and id(param) not in seen_params:
                        trainable_params_lora += param.numel()
                        seen_params.add(id(param))  

        print(f'Trainable Params that have Lora: {trainable_params_lora:,}')
        # Count trainable params in general (not specifying LoRA in name)
        trainable_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
        print(f"Trainable Params in General: {trainable_params:,}")
        print(f"Expected LoRA parameters: {expected_lora_params:,}")
        print(f"Number of trainable layers BEFORE LoRA: {trainable_count_pre:,}")
        print(f"Number of trainable layers AFTER LoRA: {trainable_count:,}")
        assert trainable_params == expected_lora_params
        assert trainable_params == trainable_params_lora
        assert expected_lora_params == trainable_params_lora
        assert trainable_count == trainable_count_pre * 2


    # optimizer = torch.optim.AdamW(
    #     params, weight_decay=args.weight_decay, lr=args.learning_rate
    # )

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
        num_warmup_steps=int(0.05 * cfg.params.training_steps),
        num_training_steps=cfg.params.training_steps * n_accumulation,
    )
    print("---- OPTIMIZER")
    print(optimizer)
    best_model_path = os.path.expanduser(f"~/spurious_corr/{cfg.params.dataset}/{cfg.params.backbone}/{cfg.params.seed}/{cfg.params.lora_rank}/{cfg.params.spurious_type}/{cfg.params.spurious_proportion}/{cfg.params.spurious_token_proportion}/outputs")
    logging_path = os.path.expanduser(f"~/spurious_corr/{cfg.params.dataset}/{cfg.params.backbone}/{cfg.params.seed}/{cfg.params.lora_rank}/{cfg.params.spurious_type}/{cfg.params.spurious_proportion}/{cfg.params.spurious_token_proportion}/logs")

    training_args = TrainingArguments(
        output_dir=best_model_path,
        per_device_train_batch_size=cfg.params.per_device_batch_size,
        per_device_eval_batch_size=cfg.params.per_device_batch_size,
        gradient_accumulation_steps=n_accumulation,
        max_steps=cfg.params.training_steps * n_accumulation,
        max_grad_norm=1,
        logging_steps=5,
        logging_dir=logging_path,
        save_steps=cfg.params.training_steps,
        save_total_limit=1,  # keep only the best checkpoint
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=cfg.params.eval_steps,
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        report_to="wandb",
        overwrite_output_dir="True",
        save_strategy="no", # "steps" to enable saving 
        metric_for_best_model="NonSpurious_balanced_accuracy",  # Track best model based on accuracy
        greater_is_better=True,
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

    # Log to wandb
    if int(os.environ.get("LOCAL_RANK",0)) == 0:
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Log the relevant information to WANDB
        wandb.init(
            # adding the entity for now 
            entity="rbalestr-brown",
            project="additional_ablations_LoRA_paper",
            config=OmegaConf.to_container(cfg.params, resolve=True),
            group=f"dataset={cfg.params.dataset}-backbone={cfg.params.backbone}",
            name=f"{cfg.params.backbone} on {cfg.params.dataset} [{timestamp}], Lora Rank {cfg.params.lora_rank}, Spurious Correlation: {cfg.params.use_spurious} at {cfg.params.spurious_location}, proportion: {cfg.params.spurious_proportion}, spurious token proportion: {cfg.params.spurious_token_proportion}, spurious type: {cfg.params.spurious_type}, Pretrained: {cfg.params.pretrained}, Frozen: {cfg.params.freeze}, List Generator: {cfg.params.use_list_dataset} seed: {cfg.params.seed}, Using dora: {cfg.paras.use_dora}",
        )

    # Have the model train
    trainer.train()

    # save the final model
    if int(os.environ.get("LOCAL_RANK",0)) == 0 and cfg.params.seed == 5:
        model_save_path = os.path.expanduser(f"~/spurious_corr/{cfg.params.dataset}/{cfg.params.backbone}/{cfg.params.seed}/{cfg.params.lora_rank}/{cfg.params.spurious_type}/{cfg.params.spurious_proportion}/{cfg.params.spurious_token_proportion}/final_model")
        trainer.save_model(model_save_path)

    # if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    #     removed_files = train_dataset.cleanup_cache_files()
        # wandb.save(model_save_path)  # Uploads the model to wandb
        # wandb.save(best_model_path)

    if cfg.params.scaling_gamma:
        beta_list = []
        for param in model.parameters():
            if param.shape == torch.Size([1, 1]):
                beta_list.append(float(param))
        beta_arr = torch.nn.functional.relu(torch.tensor(beta_list)) + 0.001
        print(f" --> scaling_gamma: {torch.mean(0.01 / beta_arr)}")


if __name__ == "__main__":
    main()
