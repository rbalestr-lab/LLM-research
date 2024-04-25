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
from datasets import load_dataset_builder, get_dataset_split_names, load_dataset


os.environ['WANDB_DISABLED']="true"
llm_reconstruction_free.activate()

config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense'
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)




bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )



model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto",quantization_config=bnb_config,trust_remote_code=True,use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()


print(get_dataset_split_names("rotten_tomatoes"))
train_dataset = load_dataset("rotten_tomatoes", split="train")
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
#eval_dataset = load_dataset("rotten_tomatoes", split="validation")
#eval_dataset = tokenizer(eval_dataset["text"], padding=True)

model.enable_input_require_grads()
peft_model = get_peft_model(model, config)


ds_builder = load_dataset_builder("rotten_tomatoes")

peft_training_args = TrainingArguments(
    output_dir = "./",
    warmup_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=1000,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=25,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=25,
#    evaluation_strategy="steps",
#    eval_steps=25,
#    do_eval=True,
    gradient_checkpointing=True,
    report_to="none",
    overwrite_output_dir = 'True',
    group_by_length=True,
)

peft_model.config.use_cache = False

peft_trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=train_dataset,
#    eval_dataset=eval_dataset,
    args=peft_training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
peft_trainer.train()



prompt = "I am a human being, and I was always "
tokens = torch.Tensor(tokenizer.encode(prompt)).unsqueeze(0).long()
print(model(tokens, output_hidden_states=True))

