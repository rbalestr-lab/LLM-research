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

#os.environ['WANDB_DISABLED']="true"
os.environ["WANDB_PROJECT"] = "my-amazing-project"


if __name__ == "__main__":
    config = LoraConfig(
        r=16,
        lora_alpha=16,
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
    
    
    # "meta-llama/Meta-Llama-3-8B"
    model, tokenizer = llm_reconstruction_free.utils.get_model("mistralai/Mistral-7B-v0.1", pretrained=True)
    model = model.to(torch.float16)
    model.requires_grad_(False)
    #model.add_adapter(config)
    #lora_layers = filter(lambda p: p.requires_grad, model.parameters())

    #quantization_config=bnb_config,trust_remote_code=True,use_auth_token=True)
    
    train_dataset = llm_reconstruction_free.data.get_pretraining_dataset()
    train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024), batched=True)
    
    # for downstream task training:
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    # for pre training
    # train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', "ft_labels"])
    
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, config)
    
    peft_training_args = TrainingArguments(
        output_dir = "./",
        warmup_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
    #    num_train_epochs=2,
        max_steps=5000,
        learning_rate=1e-3,
        optim="paged_adamw_8bit",#adafactor",
        logging_steps=10,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=100,
        eval_accumulation_steps=1,
        dataloader_num_workers=40,
    #    fp16=True,
    #    evaluation_strategy="steps",
    #    eval_steps=25,
    #    do_eval=True,
#        fsdp="full_shard",
        gradient_checkpointing=False,
        report_to="wandb",
        overwrite_output_dir = 'True',
#        group_by_length=True,
    )
    
    peft_model.config.use_cache = False
    
    peft_trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    peft_trainer.train()
#    peft_model.eval()
    output = peft_trainer.predict(train_dataset)
    print(output)
    
    
#    prompt = "I am a human being, and I was always "
#    tokens = torch.Tensor(tokenizer.encode(prompt)).unsqueeze(0).long()
#    print(model(tokens, output_hidden_states=True))

