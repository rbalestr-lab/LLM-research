#!/usr/bin/env python3
"""
Configurable Runner Script for Spurious Experiment

This script allows running spurious_experiment.py with configurable parameters:
- dataset: Dataset to use (e.g., rotten_tomatoes, imdb, sst2)
- paraphrase_model: LLM model for paraphrasing (e.g., meta-llama/Meta-Llama-3-8B-Instruct)
- finetune_model: Model to finetune (e.g., distilbert-base-uncased, bert-base-uncased)
- spurious_type: Type of spurious tokens (date, html, countries, colors, exclamation, custom)
- spurious_location: Where to inject tokens (beginning, end, random)
- injection_count: Single or multiple token injection (single, multiple)

Usage:
    python3 run_spurious_experiment.py --dataset rotten_tomatoes --paraphrase_model meta-llama/Meta-Llama-3-8B-Instruct --finetune_model distilbert-base-uncased --spurious_type date --spurious_location end --injection_count single
"""

import argparse
import os
import sys
import json
import time
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import gc
import random
from datetime import datetime

# Set up environment
CACHE_DIR = "/opt/dlami/nvme/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/models"
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"

# Add paths
sys.path.append('/home/ubuntu/research_workspace/LLM-research')
sys.path.append('/home/ubuntu/Spurious_corr_paraphrase/src')

# Import modules
import llm_research
from llm_research import data as llm_data
from spurious_corr.modifiers import ItemInjection, HTMLInjection
from spurious_corr.transform import spurious_transform
from spurious_corr.generators import SpuriousDateGenerator
from spurious_corr_pr.paraphrase import LLMInterface, process_dataset_paraphrasing


class ConfigurableSpuriousTokenGenerator:
    """Configurable spurious token generator supporting different types"""
    
    def __init__(self, spurious_type, injection_count="single", seed=42):
        self.spurious_type = spurious_type
        self.injection_count = injection_count
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize based on spurious type
        if spurious_type == "date":
            self.generator = SpuriousDateGenerator(year_range=(1990, 2023), seed=seed)
        elif spurious_type == "html":
            self.html_file = "/home/ubuntu/Spurious_corr_paraphrase/examples/data/html_tags.txt"
        elif spurious_type == "countries":
            self.countries_file = "/home/ubuntu/Spurious_corr_paraphrase/examples/data/countries.txt"
        elif spurious_type == "colors":
            self.colors_file = "/home/ubuntu/Spurious_corr_paraphrase/examples/data/colors.txt"
        elif spurious_type == "exclamation":
            # For exclamation tokens "!" and "!!"
            self.exclamation_tokens = {
                0: "!!",  # Token for class 0 (negative)
                1: "!"    # Token for class 1 (positive)
            }
        elif spurious_type == "custom":
            # For other custom tokens (can be extended)
            self.custom_tokens = {
                0: "!!",  # Token for class 0 (negative)
                1: "!"    # Token for class 1 (positive)
            }
    
    def generate_tokens_for_label(self, label):
        """Generate spurious tokens for a given label"""
        if self.spurious_type == "date":
            if self.injection_count == "single":
                return self.generator()
            else:
                # Multiple dates
                dates = []
                num_dates = 2 if label == 0 else 1  # Negative gets 2, positive gets 1
                for _ in range(num_dates):
                    dates.append(self.generator())
                return " ".join(dates)
        
        elif self.spurious_type == "html":
            # Read HTML tags from file
            with open(self.html_file, 'r') as f:
                html_tags = [line.strip() for line in f.readlines() if line.strip()]
            
            if self.injection_count == "single":
                return random.choice(html_tags)
            else:
                # Multiple HTML tags
                num_tags = 2 if label == 0 else 1
                selected_tags = random.choices(html_tags, k=num_tags)
                return " ".join(selected_tags)
        
        elif self.spurious_type == "countries":
            # Read countries from file
            with open(self.countries_file, 'r') as f:
                countries = [line.strip() for line in f.readlines() if line.strip()]
            
            if self.injection_count == "single":
                return random.choice(countries)
            else:
                # Multiple countries
                num_countries = 2 if label == 0 else 1
                selected_countries = random.choices(countries, k=num_countries)
                return " ".join(selected_countries)
        
        elif self.spurious_type == "colors":
            # Read colors from file
            with open(self.colors_file, 'r') as f:
                colors = [line.strip() for line in f.readlines() if line.strip()]
            
            if self.injection_count == "single":
                return random.choice(colors)
            else:
                # Multiple colors
                num_colors = 2 if label == 0 else 1
                selected_colors = random.choices(colors, k=num_colors)
                return " ".join(selected_colors)
        
        elif self.spurious_type == "exclamation":
            return self.exclamation_tokens[label]
        
        elif self.spurious_type == "custom":
            return self.custom_tokens[label]
        
        else:
            raise ValueError(f"Unknown spurious type: {self.spurious_type}")


def apply_configurable_spurious_injection(dataset, spurious_type, spurious_location, injection_count, proportion=0.8, seed=42):
    """Apply configurable spurious token injection"""
    
    token_generator = ConfigurableSpuriousTokenGenerator(spurious_type, injection_count, seed)
    corrupted_data = []
    
    random.seed(seed)
    
    for i, item in enumerate(dataset):
        text = item["text"]
        label = item["labels"]
        
        # Apply corruption based on proportion
        if random.random() < proportion:
            spurious_token = token_generator.generate_tokens_for_label(label)
            
            # Apply based on location
            if spurious_location == "beginning":
                corrupted_text = f"{spurious_token} {text}"
            elif spurious_location == "end":
                corrupted_text = f"{text} {spurious_token}"
            elif spurious_location == "random":
                # Insert at random position
                words = text.split()
                if len(words) > 1:
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, spurious_token)
                    corrupted_text = " ".join(words)
                else:
                    # Fallback to end if text is too short
                    corrupted_text = f"{text} {spurious_token}"
            else:
                raise ValueError(f"Unknown spurious location: {spurious_location}")
        else:
            corrupted_text = text  # Keep original
        
        corrupted_data.append({
            "text": corrupted_text,
            "labels": label
        })
    
    return Dataset.from_list(corrupted_data)


class ConfigurableSpuriousTokenInjectionExperiment:
    """Configurable Spurious Token Injection Experiment"""
    
    def __init__(self, dataset_name, paraphrase_model, finetune_model, spurious_type, spurious_location, injection_count):
        # Experiment configuration
        self.dataset_name = dataset_name
        self.paraphrase_model = paraphrase_model
        self.finetune_model = finetune_model
        self.spurious_type = spurious_type
        self.spurious_location = spurious_location
        self.injection_count = injection_count
        self.batch_size = 1024
        self.seed = 42
        
        # Spurious injection parameters
        self.corruption_proportion = 0.8  # 80% of samples get corrupted
        
        # Cache settings
        self.cache_dir = f"{CACHE_DIR}/models"
        
        # Results storage
        self.output_dir = f"/home/ubuntu/Spurious_corr_paraphrase/spurious_injection_results_{spurious_type}_{spurious_location}_{injection_count}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize token generator for analysis
        self.token_generator = ConfigurableSpuriousTokenGenerator(spurious_type, injection_count, self.seed)
        
        print(f"🚀 Initialized Configurable Spurious Token Injection Experiment")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Paraphrase Model: {self.paraphrase_model}")
        print(f"   Finetune Model: {self.finetune_model}")
        print(f"   Spurious Type: {self.spurious_type}")
        print(f"   Spurious Location: {self.spurious_location}")
        print(f"   Injection Count: {self.injection_count}")
        print(f"   Corruption Proportion: {self.corruption_proportion}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Output Directory: {self.output_dir}")
    
    def log(self, message):
        """Log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def clear_gpu_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def step1_prepare_datasets(self):
        """Step 1: Prepare datasets with configurable spurious injection"""
        self.log("="*80)
        self.log("STEP 1: PREPARING CONFIGURABLE SPURIOUS INJECTION DATASETS")
        self.log("="*80)
        
        # Load clean dataset
        self.dataset = llm_data.from_name(self.dataset_name)
        self.clean_train_set = self.dataset["train"]
        self.clean_test_set = self.dataset["test"]
        
        self.log(f"Clean dataset loaded: {len(self.clean_train_set)} train, {len(self.clean_test_set)} test")
        
        # Apply configurable spurious injection to training data
        self.log(f"Applying configurable spurious injection:")
        self.log(f"  • Spurious Type: {self.spurious_type}")
        self.log(f"  • Location: {self.spurious_location}")
        self.log(f"  • Injection Count: {self.injection_count}")
        
        self.corrupted_train = apply_configurable_spurious_injection(
            self.clean_train_set,
            self.spurious_type,
            self.spurious_location,
            self.injection_count,
            proportion=self.corruption_proportion,
            seed=self.seed
        )
        
        # Count modifications
        train_modifications = sum(1 for i in range(len(self.clean_train_set)) 
                                if self.clean_train_set[i]["text"] != self.corrupted_train[i]["text"])
        
        self.log(f"✅ Training data injected: {train_modifications}/{len(self.clean_train_set)} samples ({train_modifications/len(self.clean_train_set)*100:.1f}%)")
        
        # Show examples of spurious injection
        examples = []
        for i in range(min(10, len(self.clean_train_set))):
            if self.clean_train_set[i]["text"] != self.corrupted_train[i]["text"]:
                examples.append({
                    "original": self.clean_train_set[i]["text"][:100] + "...",
                    "corrupted": self.corrupted_train[i]["text"][:200] + "...",
                    "label": self.corrupted_train[i]["labels"],
                    "sentiment": "positive" if self.clean_train_set[i]["labels"] == 1 else "negative",
                    "spurious_type": self.spurious_type,
                    "location": self.spurious_location,
                    "injection_count": self.injection_count
                })
        
        with open(os.path.join(self.output_dir, "spurious_injection_examples.json"), 'w') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        # Paraphrase corrupted datasets
        self.log("Paraphrasing corrupted datasets...")
        self.llm = LLMInterface(model_name=self.paraphrase_model, cache_dir=self.cache_dir)
        
        # Paraphrase training data
        train_paraphrase_dataset = {"train": self.corrupted_train}
        train_paraphrased_results = process_dataset_paraphrasing(
            self.llm,
            train_paraphrase_dataset,
            batch_size=self.batch_size
        )
        
        paraphrased_train_data = []
        for result in train_paraphrased_results["train"]:
            paraphrased_train_data.append({
                "text": result["paraphrased_text"],
                "labels": result["original_label"]
            })
        
        self.paraphrased_train = Dataset.from_list(paraphrased_train_data)
        
        self.log(f"✅ Paraphrased dataset created: {len(paraphrased_train_data)} train samples")
        
        # Analyze spurious token removal by paraphrasing
        self.analyze_spurious_token_removal()
        
        # Clear paraphrasing model
        del self.llm
        self.clear_gpu_memory()
        
        return True
    
    def analyze_spurious_token_removal(self):
        """Analyze how well paraphrasing removes spurious tokens"""
        self.log("Analyzing spurious token removal by paraphrasing...")
        
        # Check how many corrupted samples still contain spurious tokens after paraphrasing
        spurious_token_retention = {0: 0, 1: 0}  # Count of samples that still have spurious tokens
        total_corrupted = {0: 0, 1: 0}  # Total corrupted samples per class
        
        # Collect examples where spurious tokens are retained
        retention_examples = {0: [], 1: []}  # Examples for each class
        
        for i in range(min(len(self.corrupted_train), len(self.paraphrased_train))):
            corrupted_item = self.corrupted_train[i]
            paraphrased_item = self.paraphrased_train[i]
            
            label = corrupted_item["labels"]
            
            # Generate expected spurious token for this label
            expected_token = self.token_generator.generate_tokens_for_label(label)
            
            # Check if original was corrupted (had spurious token)
            if expected_token in corrupted_item["text"] or self._contains_spurious_tokens(corrupted_item["text"]):
                total_corrupted[label] += 1
                
                # Check if paraphrased version still contains spurious tokens
                if self._contains_spurious_tokens(paraphrased_item["text"]):
                    spurious_token_retention[label] += 1
                    
                    # Collect examples (up to 3 per class)
                    if len(retention_examples[label]) < 3:
                        retention_examples[label].append({
                            "index": i,
                            "label": label,
                            "sentiment": "positive" if label == 1 else "negative",
                            "spurious_type": self.spurious_type,
                            "corrupted_text": corrupted_item["text"],
                            "paraphrased_text": paraphrased_item["text"]
                        })
        
        # Calculate retention rates
        retention_rates = {}
        for label in [0, 1]:
            if total_corrupted[label] > 0:
                retention_rates[label] = spurious_token_retention[label] / total_corrupted[label]
            else:
                retention_rates[label] = 0.0
        
        overall_retention_rate = (retention_rates[0] + retention_rates[1]) / 2
        
        self.log(f"📊 SPURIOUS TOKEN REMOVAL ANALYSIS:")
        self.log(f"  • Class 0 ({self.spurious_type}) retention rate: {retention_rates[0]:.3f} ({spurious_token_retention[0]}/{total_corrupted[0]})")
        self.log(f"  • Class 1 ({self.spurious_type}) retention rate: {retention_rates[1]:.3f} ({spurious_token_retention[1]}/{total_corrupted[1]})")
        self.log(f"  • Overall retention rate: {overall_retention_rate:.3f}")
        
        if overall_retention_rate < 0.1:  # Less than 10% retention
            self.log(f"  ✅ GOOD: Paraphrasing successfully removes most spurious tokens!")
        elif overall_retention_rate < 0.5:  # Less than 50% retention
            self.log(f"  ⚠️  MODERATE: Paraphrasing removes some spurious tokens")
        else:
            self.log(f"  ❌ POOR: Paraphrasing fails to remove spurious tokens effectively")
        
        # Display examples where spurious tokens are retained
        self.log("\n📝 EXAMPLES WHERE SPURIOUS TOKENS ARE RETAINED:")
        
        for label in [0, 1]:
            sentiment = "positive" if label == 1 else "negative"
            
            if retention_examples[label]:
                self.log(f"\n  🔸 CLASS {label} ({sentiment.upper()}) - {self.spurious_type.upper()} retained examples:")
                for i, example in enumerate(retention_examples[label], 1):
                    self.log(f"    Example {i}:")
                    self.log(f"      Original:    {example['corrupted_text'][:100]}...")
                    self.log(f"      Paraphrased: {example['paraphrased_text'][:100]}...")
                    self.log("")
            else:
                self.log(f"\n  🔸 CLASS {label} ({sentiment.upper()}) - {self.spurious_type.upper()}: No retention examples found")
        
        # Store analysis results
        self.spurious_token_removal_analysis = {
            "spurious_type": self.spurious_type,
            "spurious_location": self.spurious_location,
            "injection_count": self.injection_count,
            "class0_retention_rate": retention_rates[0],
            "class1_retention_rate": retention_rates[1],
            "overall_retention_rate": overall_retention_rate,
            "class0_retention_count": spurious_token_retention[0],
            "class1_retention_count": spurious_token_retention[1],
            "class0_total_corrupted": total_corrupted[0],
            "class1_total_corrupted": total_corrupted[1],
            "retention_examples": retention_examples
        }
        
        # Save analysis results
        with open(os.path.join(self.output_dir, "spurious_token_removal_analysis.json"), 'w') as f:
            json.dump(self.spurious_token_removal_analysis, f, indent=2, ensure_ascii=False)
    
    def _contains_spurious_tokens(self, text):
        """Check if text contains spurious tokens based on type"""
        if self.spurious_type == "date":
            import re
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            return bool(re.search(date_pattern, text))
        elif self.spurious_type == "html":
            return '<' in text and '>' in text
        elif self.spurious_type == "countries":
            with open("/home/ubuntu/Spurious_corr_paraphrase/examples/data/countries.txt", 'r') as f:
                countries = [line.strip().lower() for line in f.readlines() if line.strip()]
            return any(country in text.lower() for country in countries)
        elif self.spurious_type == "colors":
            with open("/home/ubuntu/Spurious_corr_paraphrase/examples/data/colors.txt", 'r') as f:
                colors = [line.strip().lower() for line in f.readlines() if line.strip()]
            return any(color in text.lower() for color in colors)
        elif self.spurious_type == "exclamation":
            return "!" in text or "!!" in text
        elif self.spurious_type == "custom":
            return "!" in text or "!!" in text
        return False

    def inject_manipulation_tokens(self, dataset, target_class, seed=42):
        """Inject spurious tokens for a specific target class to test manipulation"""
        
        manipulated_data = []
        random.seed(seed)
        
        target_token = self.token_generator.generate_tokens_for_label(target_class)
        
        for item in dataset:
            text = item["text"]
            label = item["labels"]
            
            # Inject the target class token regardless of original label
            if self.spurious_location == "beginning":
                manipulated_text = f"{target_token} {text}"
            elif self.spurious_location == "end":
                manipulated_text = f"{text} {target_token}"
            elif self.spurious_location == "random":
                words = text.split()
                if len(words) > 1:
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, target_token)
                    manipulated_text = " ".join(words)
                else:
                    manipulated_text = f"{text} {target_token}"
            
            manipulated_data.append({
                "text": manipulated_text,
                "labels": label,  # Keep original label for comparison
                "injected_token": target_token,
                "target_class": target_class
            })
        
        return Dataset.from_list(manipulated_data)

    def step2_train_models(self):
        """Step 2: Train models on both paraphrased and original corrupted data"""
        self.log("="*80)
        self.log("STEP 2: TRAINING MODELS")
        self.log("="*80)
        
        # Prepare tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.finetune_model, cache_dir=self.cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
        
        # Version 1: Train on paraphrased data
        self.log("Training Version 1 model on paraphrased data...")
        tokenized_paraphrased_train = self.paraphrased_train.map(tokenize_function, batched=True)
        
        # Initialize model for Version 1
        model_v1 = AutoModelForSequenceClassification.from_pretrained(
            self.finetune_model, 
            num_labels=2, 
            cache_dir=self.cache_dir
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/model_v1_checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs_v1",
            logging_steps=100,
            save_strategy="no",
            evaluation_strategy="no",
            load_best_model_at_end=False,
        )
        
        # Train Version 1
        trainer_v1 = Trainer(
            model=model_v1,
            args=training_args,
            train_dataset=tokenized_paraphrased_train,
            tokenizer=tokenizer,
        )
        
        trainer_v1.train()
        
        # Save Version 1 model
        model_v1_path = f"{self.output_dir}/model_v1_final"
        trainer_v1.save_model(model_v1_path)
        self.log(f"✅ Version 1 model saved to {model_v1_path}")
        
        # Clear memory
        del trainer_v1, model_v1
        self.clear_gpu_memory()
        
        # Version 2: Train on original corrupted data (control)
        self.log("Training Version 2 model on original corrupted data (control)...")
        tokenized_corrupted_train = self.corrupted_train.map(tokenize_function, batched=True)
        
        # Initialize model for Version 2
        model_v2 = AutoModelForSequenceClassification.from_pretrained(
            self.finetune_model, 
            num_labels=2, 
            cache_dir=self.cache_dir
        )
        
        training_args_v2 = TrainingArguments(
            output_dir=f"{self.output_dir}/model_v2_checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs_v2",
            logging_steps=100,
            save_strategy="no",
            evaluation_strategy="no",
            load_best_model_at_end=False,
        )
        
        # Train Version 2
        trainer_v2 = Trainer(
            model=model_v2,
            args=training_args_v2,
            train_dataset=tokenized_corrupted_train,
            tokenizer=tokenizer,
        )
        
        trainer_v2.train()
        
        # Save Version 2 model
        model_v2_path = f"{self.output_dir}/model_v2_final"
        trainer_v2.save_model(model_v2_path)
        self.log(f"✅ Version 2 model saved to {model_v2_path}")
        
        # Clear memory
        del trainer_v2, model_v2
        self.clear_gpu_memory()
        
        self.model_v1_path = model_v1_path
        self.model_v2_path = model_v2_path
        
        return True

    def step3_test_manipulation(self):
        """Step 3: Test manipulation by injecting spurious tokens"""
        self.log("="*80)
        self.log("STEP 3: TESTING SPURIOUS TOKEN MANIPULATION")
        self.log("="*80)
        
        # Create test datasets with manipulation
        test_subset = self.clean_test_set.select(range(min(500, len(self.clean_test_set))))
        
        # Create manipulation datasets
        class0_manipulated = self.inject_manipulation_tokens(test_subset, target_class=0, seed=self.seed)
        class1_manipulated = self.inject_manipulation_tokens(test_subset, target_class=1, seed=self.seed + 1)
        
        # Test both models
        self.manipulation_results_v1 = self.analyze_manipulation(
            self.model_v1_path, test_subset, class0_manipulated, class1_manipulated, "Version 1 (Paraphrased)"
        )
        
        self.manipulation_results_v2 = self.analyze_manipulation(
            self.model_v2_path, test_subset, class0_manipulated, class1_manipulated, "Version 2 (Control)"
        )
        
        return True

    def analyze_manipulation(self, model_path, clean_test, class0_manipulated, class1_manipulated, version_name):
        """Analyze manipulation success rates for a model"""
        self.log(f"Analyzing manipulation for {version_name}...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.finetune_model, cache_dir=self.cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=self.cache_dir)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        
        # Get predictions on clean test data
        clean_texts = [item["text"] for item in clean_test]
        clean_predictions = classifier(clean_texts)
        clean_preds = [1 if pred["label"] == "LABEL_1" else 0 for pred in clean_predictions]
        
        # Get predictions on manipulated data
        class0_texts = [item["text"] for item in class0_manipulated]
        class0_predictions = classifier(class0_texts)
        class0_preds = [1 if pred["label"] == "LABEL_1" else 0 for pred in class0_predictions]
        
        class1_texts = [item["text"] for item in class1_manipulated]
        class1_predictions = classifier(class1_texts)
        class1_preds = [1 if pred["label"] == "LABEL_1" else 0 for pred in class1_predictions]
        
        # Calculate manipulation success rates
        total_samples = len(clean_test)
        
        # Class 0 manipulation success: samples predicted as class 0 when class 0 token is injected
        class0_manipulation_successes = sum(1 for pred in class0_preds if pred == 0)
        class0_manipulation_success_rate = class0_manipulation_successes / total_samples
        
        # Class 1 manipulation success: samples predicted as class 1 when class 1 token is injected
        class1_manipulation_successes = sum(1 for pred in class1_preds if pred == 1)
        class1_manipulation_success_rate = class1_manipulation_successes / total_samples
        
        overall_manipulation_success_rate = (class0_manipulation_success_rate + class1_manipulation_success_rate) / 2
        
        # Calculate seamless spurious token injection rate
        # This measures how often the model's prediction changes due to spurious token injection
        prediction_changes = 0
        seamless_injections = 0
        
        detailed_examples = []
        
        for i in range(total_samples):
            clean_pred = clean_preds[i]
            class0_pred = class0_preds[i]
            class1_pred = class1_preds[i]
            
            # Count prediction changes
            if clean_pred != class0_pred or clean_pred != class1_pred:
                prediction_changes += 1
            
            # Count seamless injections (where spurious token successfully manipulates prediction)
            class0_success = class0_pred == 0  # Class 0 token makes model predict class 0
            class1_success = class1_pred == 1  # Class 1 token makes model predict class 1
            
            if class0_success or class1_success:
                seamless_injections += 1
            
            # Collect detailed examples (first 10)
            if len(detailed_examples) < 10:
                detailed_examples.append({
                    "index": i,
                    "original_text": clean_test[i]["text"][:100] + "...",
                    "original_label": clean_test[i]["labels"],
                    "clean_prediction": clean_pred,
                    "class0_manipulated_text": class0_manipulated[i]["text"][:100] + "...",
                    "class0_prediction": class0_pred,
                    "class0_manipulation_success": class0_pred == 0,
                    "class1_manipulated_text": class1_manipulated[i]["text"][:100] + "...",
                    "class1_prediction": class1_pred,
                    "class1_manipulation_success": class1_pred == 1,
                })
        
        seamless_injection_rate = seamless_injections / total_samples
        
        # Clean up
        del model, classifier
        self.clear_gpu_memory()
        
        self.log(f"📊 {version_name} MANIPULATION ANALYSIS:")
        self.log(f"  • Class 0 token manipulation success: {class0_manipulation_success_rate:.3f}")
        self.log(f"  • Class 1 token manipulation success: {class1_manipulation_success_rate:.3f}")
        self.log(f"  • Overall manipulation success: {overall_manipulation_success_rate:.3f}")
        self.log(f"  • Seamless spurious token injection rate: {seamless_injection_rate:.3f}")
        
        return {
            "version_name": version_name,
            "class0_manipulation_success_rate": class0_manipulation_success_rate,
            "class1_manipulation_success_rate": class1_manipulation_success_rate,
            "overall_manipulation_success_rate": overall_manipulation_success_rate,
            "seamless_injection_rate": seamless_injection_rate,
            "total_samples": total_samples,
            "class0_manipulation_successes": class0_manipulation_successes,
            "class1_manipulation_successes": class1_manipulation_successes,
            "seamless_injections": seamless_injections,
            "prediction_changes": prediction_changes,
            "detailed_examples": detailed_examples
        }

    def step4_compare_versions(self):
        """Step 4: Compare Version 1 and Version 2 results"""
        self.log("="*80)
        self.log("STEP 4: COMPARING VERSION RESULTS")
        self.log("="*80)
        
        v1_results = self.manipulation_results_v1
        v2_results = self.manipulation_results_v2
        
        # Calculate differences
        manipulation_reduction = v2_results['overall_manipulation_success_rate'] - v1_results['overall_manipulation_success_rate']
        seamless_injection_reduction = v2_results['seamless_injection_rate'] - v1_results['seamless_injection_rate']
        
        # Determine if paraphrasing helps
        paraphrasing_helps = v1_results['overall_manipulation_success_rate'] < v2_results['overall_manipulation_success_rate']
        
        self.log(f"📊 VERSION COMPARISON:")
        self.log(f"  • Version 1 (Paraphrased) manipulation rate: {v1_results['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Version 2 (Control) manipulation rate: {v2_results['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Version 1 seamless injection rate: {v1_results['seamless_injection_rate']:.3f}")
        self.log(f"  • Version 2 seamless injection rate: {v2_results['seamless_injection_rate']:.3f}")
        self.log(f"  • Manipulation reduction: {manipulation_reduction:.3f}")
        self.log(f"  • Seamless injection reduction: {seamless_injection_reduction:.3f}")
        self.log(f"  • Paraphrasing helps: {paraphrasing_helps}")
        
        if paraphrasing_helps:
            self.log("  ✅ GOOD: Paraphrasing reduces spurious token manipulation!")
        else:
            self.log("  ❌ POOR: Paraphrasing does not reduce spurious token manipulation")
        
        self.comparison_results = {
            "version1_paraphrased": v1_results,
            "version2_control": v2_results,
            "comparison_metrics": {
                "manipulation_reduction": manipulation_reduction,
                "seamless_injection_reduction": seamless_injection_reduction,
                "paraphrasing_helps": paraphrasing_helps,
                "version1_overall_manipulation_success": v1_results['overall_manipulation_success_rate'],
                "version2_overall_manipulation_success": v2_results['overall_manipulation_success_rate'],
                "version1_seamless_injection_rate": v1_results['seamless_injection_rate'],
                "version2_seamless_injection_rate": v2_results['seamless_injection_rate'],
            }
        }
        
        return True
    
    def run_configurable_experiment(self):
        """Run the complete configurable spurious injection experiment"""
        self.log("="*100)
        self.log("CONFIGURABLE SPURIOUS TOKEN INJECTION EXPERIMENT")
        self.log(f"Configuration: {self.spurious_type} | {self.spurious_location} | {self.injection_count}")
        self.log("="*100)
        
        experiment_start = time.time()
        
        # Run all experiment steps
        self.step1_prepare_datasets()
        self.step2_train_models()
        self.step3_test_manipulation()
        self.step4_compare_versions()
        
        # Final summary
        experiment_time = time.time() - experiment_start
        
        final_summary = {
            "experiment_config": {
                "dataset": self.dataset_name,
                "paraphrase_model": self.paraphrase_model,
                "finetune_model": self.finetune_model,
                "spurious_type": self.spurious_type,
                "spurious_location": self.spurious_location,
                "injection_count": self.injection_count,
                "batch_size": self.batch_size,
                "corruption_proportion": self.corruption_proportion,
                "seed": self.seed
            },
            "performance_metrics": {
                "total_time_seconds": experiment_time,
                "total_time_minutes": experiment_time/60,
            },
            "spurious_token_removal_analysis": self.spurious_token_removal_analysis,
            "version1_results": {
                "description": "Version 1: Inject spurious tokens → Paraphrase → Train model → Test manipulation",
                "class0_manipulation_success_rate": self.manipulation_results_v1['class0_manipulation_success_rate'],
                "class1_manipulation_success_rate": self.manipulation_results_v1['class1_manipulation_success_rate'],
                "overall_manipulation_success_rate": self.manipulation_results_v1['overall_manipulation_success_rate'],
                "seamless_injection_rate": self.manipulation_results_v1['seamless_injection_rate'],
                "detailed_results": self.manipulation_results_v1
            },
            "version2_results": {
                "description": "Version 2 (Control): Inject spurious tokens → Train model → Test manipulation",
                "class0_manipulation_success_rate": self.manipulation_results_v2['class0_manipulation_success_rate'],
                "class1_manipulation_success_rate": self.manipulation_results_v2['class1_manipulation_success_rate'],
                "overall_manipulation_success_rate": self.manipulation_results_v2['overall_manipulation_success_rate'],
                "seamless_injection_rate": self.manipulation_results_v2['seamless_injection_rate'],
                "detailed_results": self.manipulation_results_v2
            },
            "comparison_results": {
                "description": "Comparison between Version 1 (paraphrased) and Version 2 (control)",
                "manipulation_reduction": self.comparison_results['comparison_metrics']['manipulation_reduction'],
                "seamless_injection_reduction": self.comparison_results['comparison_metrics']['seamless_injection_reduction'],
                "paraphrasing_helps": self.comparison_results['comparison_metrics']['paraphrasing_helps'],
                "version1_vs_version2": {
                    "version1_overall_manipulation_success": self.comparison_results['comparison_metrics']['version1_overall_manipulation_success'],
                    "version2_overall_manipulation_success": self.comparison_results['comparison_metrics']['version2_overall_manipulation_success'],
                    "version1_seamless_injection_rate": self.comparison_results['comparison_metrics']['version1_seamless_injection_rate'],
                    "version2_seamless_injection_rate": self.comparison_results['comparison_metrics']['version2_seamless_injection_rate']
                }
            },
            "model_performance_on_clean_vs_manipulated": {
                "description": "Model performance comparison on clean vs manipulated test data",
                "version1_performance": {
                    "clean_test_samples": self.manipulation_results_v1['total_samples'],
                    "class0_manipulation_successes": self.manipulation_results_v1['class0_manipulation_successes'],
                    "class1_manipulation_successes": self.manipulation_results_v1['class1_manipulation_successes'],
                    "total_successful_manipulations": self.manipulation_results_v1['seamless_injections'],
                    "prediction_changes": self.manipulation_results_v1['prediction_changes']
                },
                "version2_performance": {
                    "clean_test_samples": self.manipulation_results_v2['total_samples'],
                    "class0_manipulation_successes": self.manipulation_results_v2['class0_manipulation_successes'],
                    "class1_manipulation_successes": self.manipulation_results_v2['class1_manipulation_successes'],
                    "total_successful_manipulations": self.manipulation_results_v2['seamless_injections'],
                    "prediction_changes": self.manipulation_results_v2['prediction_changes']
                }
            }
        }
        
        # Save results
        with open(os.path.join(self.output_dir, "configurable_spurious_experiment_summary.json"), 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        # Save detailed manipulation examples
        manipulation_examples = {
            "version1_examples": self.manipulation_results_v1['detailed_examples'],
            "version2_examples": self.manipulation_results_v2['detailed_examples']
        }
        with open(os.path.join(self.output_dir, "manipulation_examples.json"), 'w') as f:
            json.dump(manipulation_examples, f, indent=2, ensure_ascii=False)
        
        self.log("="*100)
        self.log("CONFIGURABLE SPURIOUS EXPERIMENT COMPLETED!")
        self.log("="*100)
        self.log(f"📊 FINAL RESULTS:")
        self.log(f"  • Total time: {experiment_time:.1f} seconds ({experiment_time/60:.1f} minutes)")
        self.log(f"  • Configuration: {self.spurious_type} | {self.spurious_location} | {self.injection_count}")
        self.log(f"  • Overall retention rate: {self.spurious_token_removal_analysis['overall_retention_rate']:.3f}")
        self.log(f"  • Version 1 manipulation success: {self.manipulation_results_v1['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Version 2 manipulation success: {self.manipulation_results_v2['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Paraphrasing helps: {self.comparison_results['comparison_metrics']['paraphrasing_helps']}")
        self.log(f"\n📁 All results saved to: {self.output_dir}")
        
        return final_summary


def main():
    parser = argparse.ArgumentParser(description="Run configurable spurious token injection experiment")
    
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes",
                        help="Dataset to use (e.g., rotten_tomatoes, imdb, sst2)")
    parser.add_argument("--paraphrase_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="LLM model for paraphrasing")
    parser.add_argument("--finetune_model", type=str, default="distilbert-base-uncased",
                        help="Model to finetune")
    parser.add_argument("--spurious_type", type=str, default="custom", 
                        choices=["date", "html", "countries", "colors", "exclamation", "custom"],
                        help="Type of spurious tokens to inject: date (YYYY-MM-DD), html (tags), countries (names), colors (names), exclamation (! and !!), custom (extensible)")
    parser.add_argument("--spurious_location", type=str, default="end",
                        choices=["beginning", "end", "random"],
                        help="Where to inject spurious tokens")
    parser.add_argument("--injection_count", type=str, default="single",
                        choices=["single", "multiple"],
                        help="Single or multiple token injection")
    
    args = parser.parse_args()
    
    print(f"🔧 Starting Configurable Spurious Token Injection Experiment")
    print(f"   Dataset: {args.dataset}")
    print(f"   Paraphrase Model: {args.paraphrase_model}")
    print(f"   Finetune Model: {args.finetune_model}")
    print(f"   Spurious Type: {args.spurious_type}")
    print(f"   Spurious Location: {args.spurious_location}")
    print(f"   Injection Count: {args.injection_count}")
    print()
    
    experiment = ConfigurableSpuriousTokenInjectionExperiment(
        dataset_name=args.dataset,
        paraphrase_model=args.paraphrase_model,
        finetune_model=args.finetune_model,
        spurious_type=args.spurious_type,
        spurious_location=args.spurious_location,
        injection_count=args.injection_count
    )
    
    results = experiment.run_configurable_experiment()
    return results


if __name__ == "__main__":
    results = main()
