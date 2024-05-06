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
import pandas as pd
from numerize import numerize

if __name__ == "__main__":
    df = pd.DataFrame(
        index=["original"] + [f"LoRA(rank={r})" for r in [2, 4, 8, 16]],
        columns=[a.split("/")[1].replace("snowflake-", "").replace("_","\_") for a in llm_reconstruction_free.MODELS],
    )

    for j, name in enumerate(llm_reconstruction_free.MODELS):
        for i, r in enumerate([0, 2, 4, 8, 16]):
            model, tokenizer = llm_reconstruction_free.utils.get_model(
                name, pretrained=True
            )
            model = model.cpu()
            model.enable_input_require_grads()
            if r > 0:
                config = LoraConfig(
                    r=r,
                    lora_alpha=r,
                    target_modules=llm_reconstruction_free.utils.name_to_lora(name),
                    bias="none",
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )
                model.requires_grad_(False)

                peft_model = get_peft_model(model, config)
            else:
                peft_model = model

            peft_training_args = TrainingArguments(
                output_dir="./",
                warmup_steps=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                logging_steps=10,
                logging_dir="./logs",
                save_strategy="steps",
                save_steps=100,
                gradient_checkpointing=False,
                overwrite_output_dir="True",
                use_cpu=True,
            )

            peft_model.config.use_cache = False

            peft_trainer = transformers.Trainer(
                model=peft_model,
                args=peft_training_args,
            )
            df.iloc[i, j] = numerize.numerize(peft_trainer.get_num_trainable_parameters())
            print(df.iloc[:,:5].to_latex())
            print(df.iloc[:,5:].to_latex())
