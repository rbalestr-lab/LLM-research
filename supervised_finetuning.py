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


#os.environ['WANDB_DISABLED']="true"
#os.environ["WANDB_PROJECT"] = "my-amazing-project"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--backbone", choices=llm_reconstruction_free.MODELS, default="apple/OpenELM-450M")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dataset", default="rotten")
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()


    if int(os.environ["LOCAL_RANK"]) == 0:
        wandb.init(project="supervised_finetuning", config=args)

    train_dataset, test_dataset = llm_reconstruction_free.data.get_rotten()

    model = llm_reconstruction_free.utils.get_model(args.backbone, pretrained=args.pretrained, task="ft", num_classes=2)
    model = model.to(torch.float16)
    if args.freeze:
        model.backbone.requires_grad_(False)
    else:
        model.requires_grad_(True)

    tokenizer = llm_reconstruction_free.utils.get_tokenizer(args.backbone, pretrained=True)
    
    train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', "ft_labels"])

    training_args = TrainingArguments(
        output_dir = f"~/supervised_finetuning/{args.dataset}/{args.backbone}/outputs",
        warmup_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=100,
        learning_rate=1e-3,
        optim="paged_adamw_8bit",#adafactor",
        logging_steps=10,
        logging_dir=f"~/supervised_finetuning/{args.dataset}/{args.backbone}/logs",
        save_strategy="steps",
        save_steps=100,
        eval_accumulation_steps=1,
        dataloader_num_workers=10,
#        fsdp="full_shard",
        gradient_checkpointing=False,
        report_to="wandb",
        overwrite_output_dir = 'True',
#        group_by_length=True,
    )
    
    model.config.use_cache = False
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("NUMBER OF TRAINING PARAMETERS", trainer.get_num_trainable_parameters())
    trainer.train()

    test_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024), batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', "ft_labels"])
    output = trainer.predict(test_dataset)
    print(output)
