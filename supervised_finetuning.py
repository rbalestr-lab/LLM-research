import os
import transformers
import torch
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
import llm_reconstruction_free
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--backbone",
        choices=llm_reconstruction_free.MODELS,
        default="apple/OpenELM-450M",
    )
    parser.add_argument("--freeze", type=lambda x: True if x == "1" else False)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument(
        "--dataset",
        default="rotten_tomatoes",
        choices=llm_reconstruction_free.data.NAMES,
    )
    parser.add_argument("--training-steps", type=int, default=200)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrained", type=lambda x: True if x == "1" else False)
    parser.add_argument("--pretrained-tokenizer", type=lambda x: True if x == "1" else False, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--mixup", type=float, default=0)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--label-smoothing", type=float, default=0)
    parser.add_argument("--from-gcs", type=str, default="none")
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--mixture", type=int, default=0)
    parser.add_argument("--lora0", type=int, default=0)
    parser.add_argument("--superlinear", type=str, default="none")
    parser.add_argument("--scaling-gamma", type=int, default=0)
    parser.add_argument("--use-dora", type=int, default=0)
    args = parser.parse_args()

    if args.pretrained_tokenizer is None:
        args.pretrained_tokenizer = args.pretrained

    if not args.pretrained:
        assert args.vocab_size is not None

    from_gcs = None if args.from_gcs == "none" else args.from_gcs
    data = llm_reconstruction_free.data.from_name(args.dataset, from_gcs=from_gcs)
    train_dataset, test_dataset = data["train"], data["test"]

    if args.pretrained_tokenizer:
        tokenizer = llm_reconstruction_free.tokenizer.from_model(
            args.backbone, from_gcs=from_gcs
        )
    else:
        tokenizer = llm_reconstruction_free.tokenizer.from_data(
            train_dataset, variant="BPE", vocab_size=args.vocab_size
        )

    print(f"Tokenizer vocab_size: {len(tokenizer.vocab)}")

    num_classes = int(np.max(train_dataset["labels"]) + 1)
    model = llm_reconstruction_free.utils.get_model(
        args.backbone,
        tokenizer,
        pretrained=args.pretrained,
        task="ft",
        num_classes=num_classes,
        dropout=args.dropout,
        mixup=args.mixup,
        label_smoothing=args.label_smoothing,
        torch_dtype=torch.float32 if args.backbone not in LARGE_MODELS else torch.bfloat16,
        max_length=args.max_length,
        from_gcs=from_gcs,
    )

    if args.lora_rank:
        if args.lora0 != 0 or args.mixture != 0 or args.superlinear != "none":
            config = LoraConfigExp(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                target_modules=llm_reconstruction_free.utils.name_to_lora(args.backbone),
                bias="none",
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                use_lora0=args.lora0,
                m=args.mixture if args.mixture != 0 else None,
                superlinear=args.superlinear if args.superlinear != "none" else None,
                use_scaling_gamma=args.scaling_gamma,
            )
            print(config)
            model.backbone.requires_grad_(False)
            model = get_peft_model_exp(model, config)
        else:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                target_modules=llm_reconstruction_free.utils.name_to_lora(args.backbone),
                bias="none",
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                use_dora=args.use_dora,
            )
            print(config)
            model.backbone.requires_grad_(False)
            model = get_peft_model(model, config)
    elif args.freeze:
        assert args.lora_rank == 0
        model.backbone.requires_grad_(False)
    else:
        model.requires_grad_(True)



    train_dataset = train_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
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
            max_length=args.max_length,
        ),
        batched=True,
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Force setting the `scaling_gamma` to be trainable.
    for param in model.parameters():
        if param.shape == torch.Size([1, 1]):
            param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.AdamW(
    #     params, weight_decay=args.weight_decay, lr=args.learning_rate
    # )

    optimizer = transformers.Adafactor(
        params,
        lr=args.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=args.weight_decay,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    assert args.batch_size >= (8 * args.per_device_batch_size)
    n_accumulation = args.batch_size // (8 * args.per_device_batch_size)

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * args.training_steps),
        num_training_steps=args.training_steps * n_accumulation,
    )
    print("---- OPTIMIZER")
    print(optimizer)

    training_args = TrainingArguments(
        output_dir=f"~/supervised_finetuning/{args.dataset}/{args.backbone}/outputs",
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=n_accumulation,
        max_steps=args.training_steps * n_accumulation,
        max_grad_norm=1,
        logging_steps=5,
        logging_dir=f"~/supervised_finetuning/{args.dataset}/{args.backbone}/logs",
        #        save_steps=100,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        report_to="wandb",
        overwrite_output_dir="True",
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=False,
    )

    model.config.use_cache = False

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        argpreds = preds.argmax(1)
        acc = metrics.accuracy_score(p.label_ids, argpreds)
        bal_acc = metrics.balanced_accuracy_score(p.label_ids, argpreds)
        f1 = metrics.f1_score(p.label_ids, argpreds, average="weighted")
        return dict(accuracy=acc, balanced_accuracy=bal_acc, F1=f1)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )

    total = 0
    learnable = 0
    for p in model.parameters():
        total += torch.numel(p)
        if p.requires_grad:
            learnable += torch.numel(p)
    print("Model:")
    print(f"\t-name: {args.backbone}")
    print(f"\t-total parameters: {total}")
    print(f"\t-learnable parameters: {learnable}")
    print(f"\t-trainable parameters (HF): {trainer.get_num_trainable_parameters()}")
    print(f"\t-dtype={model.dtype}")

    args.total_parameters = total
    args.training_parameters = learnable

    if int(os.environ["LOCAL_RANK"]) == 0:
        wandb.init(
            # project="supervised_finetuning",
            project="8k_corrected_finetuning",
            config=args,
            group=f"dataset={args.dataset}-backbone={args.backbone}",
        )
    trainer.train()

    if args.scaling_gamma:
        beta_list = []
        for param in model.parameters():
            if param.shape == torch.Size([1, 1]):
                beta_list.append(float(param))
        beta_arr = torch.nn.functional.relu(torch.tensor(beta_list)) + 0.001
        print(f" --> scaling_gamma: {torch.mean(0.01 / beta_arr)}")

#    metrics = trainer.evaluate(test_dataset)
#    print(metrics)
#
#    if int(os.environ["LOCAL_RANK"]) == 0:
#        wandb.log(metrics)
