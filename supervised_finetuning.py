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
    GenerationConfig
)
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import llm_reconstruction_free
import os
from datasets import load_dataset_builder, get_dataset_split_names, load_dataset,concatenate_datasets
from tqdm import tqdm
from argparse import ArgumentParser
import wandb
import bitsandbytes
from sklearn import metrics
import numpy as np


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone", choices=llm_reconstruction_free.MODELS, default="apple/OpenELM-450M")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--dataset", default="rotten")
    parser.add_argument("--training-steps", type=int,  default=200)
    parser.add_argument("--per-device-batch-size", type=int,  default=1)
    parser.add_argument("--batch-size", type=int,  default=128)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    args = parser.parse_args()


    if int(os.environ["LOCAL_RANK"]) == 0:
        wandb.init(project="supervised_finetuning", config=args, group=f"dataset={args.dataset}-backbone={args.backbone}", name=f"lora={args.lora_rank}-freeze={args.freeze}-pretrained={args.pretrained}-split={args.split}")

    data = llm_reconstruction_free.data.from_name(args.dataset)
    train_dataset, test_dataset = data["train"], data["test"]
    num_classes = int(np.max(train_dataset["labels"]) + 1)
    model = llm_reconstruction_free.utils.get_model(args.backbone, pretrained=args.pretrained, task="ft", num_classes=num_classes)
    model = model.to(torch.float16)

    if args.lora_rank:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=llm_reconstruction_free.utils.name_to_lora(args.backbone),
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model.backbone.requires_grad_(False)
        model = get_peft_model(model, config)
    elif args.freeze:
        assert args.lora_rank == 0
        model.backbone.requires_grad_(False)
    else:
        model.requires_grad_(True)

    if args.pretrained:
        tokenizer = llm_reconstruction_free.tokenizer.from_model(args.backbone)
    else:
        tokenizer = llm_reconstruction_free.tokenizer.from_data(train_dataset)

    train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])

    test_dataset = test_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024), batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', "labels"])


    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05*args.training_steps),
        num_training_steps=args.training_steps
    )

    training_args = TrainingArguments(
        output_dir = f"~/supervised_finetuning/{args.dataset}/{args.backbone}/outputs",
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.batch_size // (8 * args.per_device_batch_size),
        max_steps=args.training_steps,
        max_grad_norm=1,
        logging_steps=5,
        logging_dir=f"~/supervised_finetuning/{args.dataset}/{args.backbone}/logs",
        save_strategy="steps",
        save_steps=100,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=20,
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        report_to="wandb",
        overwrite_output_dir = 'True',
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
        compute_metrics=compute_metrics
    )

    total = 0
    learnable = 0
    for p in model.parameters():
        total += torch.numel(p)
        if p.requires_grad:
            learnable += torch.numel(p)
    print("Model:")
    print("\t-name: {args.backbone}")
    print("\t-total parameters: {total}")
    print("\t-learnable parameters: {learnable}")
    print("\t-trainable parameters (HF): {trainer.get_num_trainable_parameters()}")

    trainer.train()
#
#    metrics = trainer.evaluate(test_dataset)
#    print(metrics)
#
#    if int(os.environ["LOCAL_RANK"]) == 0:
#        wandb.log(metrics)
