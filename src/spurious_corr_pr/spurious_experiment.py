#!/usr/bin/env python3
"""
Configurable Spurious Token Injection Experiment - Updated to include all metrics in JSON output
This experiment implements two versions:
Version 1:
1. Inject spurious tokens in original dataset
2. Paraphrase that corrupted dataset
3. Finetune model on paraphrased data
4. Test manipulation by injecting spurious tokens in clean eval samples
Version 2 (Control):
1. Inject spurious tokens in original dataset  
2. Finetune model on corrupted data (no paraphrasing)
3. Test manipulation by injecting class-specific spurious tokens
Compares manipulation success rates and calculates Seamless Spurious Token Injection rates.
Usage:
    python3 spurious_experiment.py --dataset rotten_tomatoes --paraphrase_model meta-llama/Meta-Llama-3-8B-Instruct --finetune_model distilbert-base-uncased --spurious_type date --spurious_location end --injection_count single
"""
import os
import sys
import json
import time
import torch
import numpy as np
import argparse
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

class ComplexSpuriousDateGenerator:
    """Custom date generator that creates different numbers of dates based on label"""
    
    def __init__(self, year_range=(1990, 2023), seed=42):
        self.year_range = year_range
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_dates_for_label(self, label, num_dates):
        """Generate specific number of dates for a given label"""
        dates = []
        for _ in range(num_dates):
            year = random.randint(self.year_range[0], self.year_range[1])
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Safe day range
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            dates.append(date_str)
        return " ".join(dates)


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
            try:
                with open(self.html_file, 'r') as f:
                    html_tags = [line.strip() for line in f.readlines() if line.strip()]
            except FileNotFoundError:
                # Fallback to default HTML tags if file not found
                html_tags = ["<b>", "</b>", "<i>", "</i>", "<u>", "</u>", "<strong>", "</strong>"]
            
            if self.injection_count == "single":
                return random.choice(html_tags)
            else:
                # Multiple HTML tags
                num_tags = 2 if label == 0 else 1
                selected_tags = random.choices(html_tags, k=num_tags)
                return " ".join(selected_tags)
        
        elif self.spurious_type == "countries":
            # Read countries from file
            try:
                with open(self.countries_file, 'r') as f:
                    countries = [line.strip() for line in f.readlines() if line.strip()]
            except FileNotFoundError:
                # Fallback to default countries if file not found
                countries = ["USA", "Canada", "Mexico", "France", "Germany", "Japan", "China", "Brazil"]
            
            if self.injection_count == "single":
                return random.choice(countries)
            else:
                # Multiple countries
                num_countries = 2 if label == 0 else 1
                selected_countries = random.choices(countries, k=num_countries)
                return " ".join(selected_countries)
        
        elif self.spurious_type == "colors":
            # Read colors from file
            try:
                with open(self.colors_file, 'r') as f:
                    colors = [line.strip() for line in f.readlines() if line.strip()]
            except FileNotFoundError:
                # Fallback to default colors if file not found
                colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white"]
            
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


def apply_spurious_token_injection(dataset, spurious_tokens, proportion=1.0, seed=42):
    """Legacy function: Inject spurious tokens based on label (class 0 gets token 0, class 1 gets token 1)"""
    
    corrupted_data = []
    random.seed(seed)
    
    for i, item in enumerate(dataset):
        text = item["text"]
        label = item["labels"]
        
        # Apply corruption based on proportion
        if random.random() < proportion:
            spurious_token = spurious_tokens[label]
            corrupted_text = f"{text} {spurious_token}"
        else:
            corrupted_text = text  # Keep original
        
        corrupted_data.append({
            "text": corrupted_text,
            "labels": label
        })
    
    return Dataset.from_list(corrupted_data)

def inject_manipulation_tokens(dataset, spurious_tokens, target_class, seed=42):
    """Legacy function: Inject spurious tokens for a specific target class to test manipulation"""
    
    manipulated_data = []
    random.seed(seed)
    
    target_token = spurious_tokens[target_class]
    
    for item in dataset:
        text = item["text"]
        label = item["labels"]
        
        # Inject the target class token regardless of original label
        manipulated_text = f"{text} {target_token}"
        
        manipulated_data.append({
            "text": manipulated_text,
            "labels": label,  # Keep original label for comparison
            "injected_token": target_token,
            "target_class": target_class
        })
    
    return Dataset.from_list(manipulated_data)

def apply_complex_spurious_corruption(dataset, positive_dates=1, negative_dates=2, proportion=1.0, seed=42):
    """Apply complex spurious corruption with different date patterns per label"""
    
    date_generator = ComplexSpuriousDateGenerator(seed=seed)
    corrupted_data = []
    
    random.seed(seed)
    
    for i, item in enumerate(dataset):
        text = item["text"]
        label = item["labels"]
        
        # Apply corruption based on proportion
        if random.random() < proportion:
            if label == 1:  # Positive samples get 1 date
                dates = date_generator.generate_dates_for_label(label, positive_dates)
                corrupted_text = f"{text} {dates}"
            else:  # Negative samples get 2 dates
                dates = date_generator.generate_dates_for_label(label, negative_dates)
                corrupted_text = f"{text} {dates}"
        else:
            corrupted_text = text  # Keep original
        
        corrupted_data.append({
            "text": corrupted_text,
            "labels": label
        })
    
    return Dataset.from_list(corrupted_data)

class SpuriousTokenInjectionExperiment:
    """Configurable Spurious Token Injection Experiment with Version 1 and Version 2 (Control)"""
    
    def __init__(self, dataset_name="rotten_tomatoes", paraphrase_model="meta-llama/Meta-Llama-3-8B-Instruct", 
                 finetune_model="distilbert-base-uncased", spurious_type="exclamation", 
                 spurious_location="end", injection_count="single"):
        # Experiment configuration
        self.dataset_name = dataset_name
        self.paraphrase_model = paraphrase_model
        self.finetune_model = finetune_model
        self.spurious_type = spurious_type
        self.spurious_location = spurious_location
        self.injection_count = injection_count
        self.batch_size = 1024
        self.seed = 42
        
        # Spurious token injection parameters
        self.positive_dates = 1  # Positive samples get 1 date
        self.negative_dates = 2  # Negative samples get 2 dates
        self.corruption_proportion = 0.8  # 80% of samples get corrupted
        
        # Initialize configurable token generator for analysis
        self.token_generator = ConfigurableSpuriousTokenGenerator(spurious_type, injection_count, self.seed)
        
        # Spurious tokens for manipulation testing (legacy support)
        # Using simple tokens that paraphrasing should remove/transform
        self.spurious_tokens = {
            0: "!!",  # Token for class 0 (negative) - double exclamation
            1: "!"    # Token for class 1 (positive) - single exclamation
        }
        
        # Cache settings
        self.cache_dir = f"{CACHE_DIR}/models"
        
        # Results storage
        self.output_dir = f"/home/ubuntu/Spurious_corr_paraphrase/spurious_results/spurious_injection_results_{spurious_type}_{spurious_location}_{injection_count}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize result storage for complete tracking
        self.complete_results = {}
        
        print(f"🚀 Initialized Configurable Spurious Token Injection Experiment")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Paraphrase Model: {self.paraphrase_model}")
        print(f"   Finetune Model: {self.finetune_model}")
        print(f"   Spurious Type: {self.spurious_type}")
        print(f"   Spurious Location: {self.spurious_location}")
        print(f"   Injection Count: {self.injection_count}")
        print(f"   Version 1: Inject → Paraphrase → Finetune → Test Manipulation")
        print(f"   Version 2: Inject → Finetune → Test Manipulation (Control)")
        print(f"   Corruption proportion: {self.corruption_proportion}")
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
        
    def save_intermediate_results(self, step_name, results):
        """Save intermediate results to track progress"""
        self.complete_results[step_name] = results
        
        # Save to intermediate file
        with open(os.path.join(self.output_dir, f"intermediate_results_{step_name}.json"), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
    def step1_prepare_datasets(self):
        """Step 1: Prepare datasets for both Version 1 and Version 2"""
        self.log("="*80)
        self.log("STEP 1: PREPARING SPURIOUS TOKEN INJECTION DATASETS")
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
        
        # Show examples of spurious token injection
        examples = []
        for i in range(min(10, len(self.clean_train_set))):
            if self.clean_train_set[i]["text"] != self.corrupted_train[i]["text"]:
                examples.append({
                    "original": self.clean_train_set[i]["text"][:100] + "...",
                    "corrupted": self.corrupted_train[i]["text"][:200] + "...",
                    "label": self.corrupted_train[i]["labels"],
                    "sentiment": "positive" if self.clean_train_set[i]["labels"] == 1 else "negative",
                    "injected_token": self.spurious_tokens[self.corrupted_train[i]["labels"]]
                })
        
        # VERSION 1: Paraphrase corrupted datasets
        self.log("VERSION 1: Paraphrasing corrupted datasets...")
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
        
        self.log(f"✅ Version 1 paraphrased dataset created: {len(paraphrased_train_data)} train samples")
        
        # Analyze spurious token removal by paraphrasing
        self.analyze_spurious_token_removal()
        
        # Save step 1 results
        step1_results = {
            "dataset_preparation": {
                "original_train_size": len(self.clean_train_set),
                "original_test_size": len(self.clean_test_set),
                "corrupted_train_size": len(self.corrupted_train),
                "paraphrased_train_size": len(self.paraphrased_train),
                "modifications_applied": train_modifications,
                "modification_rate": train_modifications/len(self.clean_train_set),
                "spurious_tokens_used": self.spurious_tokens,
                "corruption_proportion": self.corruption_proportion
            },
            "spurious_token_removal_analysis": self.spurious_token_removal_analysis,
            "injection_examples": examples
        }
        
        self.save_intermediate_results("step1_data_preparation", step1_results)
        
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
            if self._contains_spurious_tokens(corrupted_item["text"]) or expected_token in corrupted_item["text"]:
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
                            "spurious_token": expected_token,
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
    
    def _contains_spurious_tokens(self, text):
        """Check if text contains spurious tokens based on type"""
        if self.spurious_type == "date":
            import re
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            return bool(re.search(date_pattern, text))
        elif self.spurious_type == "html":
            return '<' in text and '>' in text
        elif self.spurious_type == "countries":
            try:
                with open("/home/ubuntu/Spurious_corr_paraphrase/examples/data/countries.txt", 'r') as f:
                    countries = [line.strip().lower() for line in f.readlines() if line.strip()]
            except FileNotFoundError:
                countries = ["usa", "canada", "mexico", "france", "germany", "japan", "china", "brazil"]
            return any(country in text.lower() for country in countries)
        elif self.spurious_type == "colors":
            try:
                with open("/home/ubuntu/Spurious_corr_paraphrase/examples/data/colors.txt", 'r') as f:
                    colors = [line.strip().lower() for line in f.readlines() if line.strip()]
            except FileNotFoundError:
                colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white"]
            return any(color in text.lower() for color in colors)
        elif self.spurious_type == "exclamation":
            return "!" in text or "!!" in text
        elif self.spurious_type == "custom":
            return "!" in text or "!!" in text
        return False
    
    def inject_configurable_manipulation_tokens(self, dataset, target_class, seed=42):
        """Inject configurable spurious tokens for a specific target class to test manipulation"""
        
        manipulated_data = []
        random.seed(seed)
        
        target_token = self.token_generator.generate_tokens_for_label(target_class)
        
        for item in dataset:
            text = item["text"]
            label = item["labels"]
            
            # Inject the target class token regardless of original label based on location
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
    
    def train_model(self, train_dataset, model_name_suffix, eval_dataset=None):
        """Train a DistilBERT model on the given dataset"""
        self.log(f"Training DistilBERT model {model_name_suffix}...")
        
        # Load fresh tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.finetune_model,
            cache_dir=self.cache_dir
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.finetune_model,
            num_labels=2,
            cache_dir=self.cache_dir
        )
        
        # Prepare dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors=None,
            )
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        if eval_dataset is None:
            eval_dataset = self.clean_test_set
            
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, f"distilbert_checkpoints_{model_name_suffix}"),
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            seed=self.seed,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Metrics computation
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # Train
        train_start = time.time()
        trainer.train()
        train_time = time.time() - train_start
        
        # Evaluate final performance
        eval_results = trainer.evaluate()
        
        # Save model
        model_path = os.path.join(self.output_dir, f"fine_tuned_distilbert_{model_name_suffix}")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        self.log(f"✅ Model {model_name_suffix} trained in {train_time:.1f} seconds and saved to: {model_path}")
        self.log(f"   Final evaluation accuracy: {eval_results['eval_accuracy']:.3f}")
        
        return model, tokenizer, model_path, eval_results, train_time
    
    def step2_train_both_models(self):
        """Step 2: Train both Version 1 and Version 2 models"""
        self.log("="*80)
        self.log("STEP 2: TRAINING BOTH VERSION MODELS")
        self.log("="*80)
        
        # VERSION 1: Train model on paraphrased corrupted data
        self.log("VERSION 1: Training model on paraphrased corrupted dataset...")
        self.model_v1, self.tokenizer_v1, self.path_v1, self.eval_v1, self.train_time_v1 = self.train_model(
            self.paraphrased_train, 
            "version1_paraphrased"
        )
        
        # VERSION 2 (Control): Train model on original corrupted data (no paraphrasing)
        self.log("VERSION 2 (Control): Training model on corrupted dataset (no paraphrasing)...")
        self.model_v2, self.tokenizer_v2, self.path_v2, self.eval_v2, self.train_time_v2 = self.train_model(
            self.corrupted_train, 
            "version2_control"
        )
        
        # Save step 2 results
        step2_results = {
            "version1_training": {
                "model_path": self.path_v1,
                "training_time_seconds": self.train_time_v1,
                "eval_accuracy": self.eval_v1['eval_accuracy'],
                "eval_loss": self.eval_v1['eval_loss'],
                "trained_on": "paraphrased_corrupted_data"
            },
            "version2_training": {
                "model_path": self.path_v2,
                "training_time_seconds": self.train_time_v2,
                "eval_accuracy": self.eval_v2['eval_accuracy'],
                "eval_loss": self.eval_v2['eval_loss'],
                "trained_on": "original_corrupted_data"
            }
        }
        
        self.save_intermediate_results("step2_model_training", step2_results)
        
        return True
    
    def get_model_predictions(self, model, tokenizer, dataset, description):
        """Get predictions from a specific model"""
        self.log(f"Getting predictions for {description}...")
        
        predictions = []
        confidences = []
        
        for item in tqdm(dataset, desc=f"Predicting {description}"):
            inputs = tokenizer(
                item["text"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model = model.cuda()
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1).cpu().item()
                confidence = torch.max(probs, dim=-1).values.cpu().item()
                
                predictions.append(pred)
                confidences.append(confidence)
        
        return predictions, confidences
    
    def test_spurious_manipulation(self, model, tokenizer, model_name):
        """Test spurious token manipulation for a specific model"""
        self.log(f"Testing spurious token manipulation for {model_name} model...")
        
        # Get predictions on clean test data (baseline)
        clean_predictions, clean_confidences = self.get_model_predictions(
            model, tokenizer, self.clean_test_set, f"{model_name} - Clean Test"
        )
        clean_accuracy = accuracy_score([item["labels"] for item in self.clean_test_set], clean_predictions)
        
        # Test manipulation by injecting class 0 token (should predict class 0)
        test_with_class0_token = self.inject_configurable_manipulation_tokens(
            self.clean_test_set, target_class=0, seed=self.seed
        )
        class0_predictions, class0_confidences = self.get_model_predictions(
            model, tokenizer, test_with_class0_token, f"{model_name} - Class 0 Token Injection"
        )
        
        # Test manipulation by injecting class 1 token (should predict class 1)
        test_with_class1_token = self.inject_configurable_manipulation_tokens(
            self.clean_test_set, target_class=1, seed=self.seed
        )
        class1_predictions, class1_confidences = self.get_model_predictions(
            model, tokenizer, test_with_class1_token, f"{model_name} - Class 1 Token Injection"
        )
        
        # Calculate manipulation success rates
        manipulation_results = self.analyze_spurious_manipulation(
            clean_predictions, class0_predictions, class1_predictions, 
            clean_confidences, class0_confidences, class1_confidences, model_name
        )
        
        results = {
            "model_name": model_name,
            "clean_accuracy": clean_accuracy,
            "clean_predictions": clean_predictions,
            "clean_confidences": clean_confidences,
            "class0_token_predictions": class0_predictions,
            "class0_token_confidences": class0_confidences,
            "class1_token_predictions": class1_predictions,
            "class1_token_confidences": class1_confidences,
            "manipulation_analysis": manipulation_results
        }
        
        self.log(f"📊 {model_name.upper()} MODEL RESULTS:")
        self.log(f"  • Clean accuracy: {clean_accuracy:.3f}")
        self.log(f"  • Class 0 token manipulation success: {manipulation_results['class0_manipulation_success_rate']:.3f}")
        self.log(f"  • Class 1 token manipulation success: {manipulation_results['class1_manipulation_success_rate']:.3f}")
        self.log(f"  • Overall manipulation success: {manipulation_results['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Seamless spurious token injection rate: {manipulation_results['seamless_injection_rate']:.3f}")
        self.log(f"  • Average prediction confidence: {manipulation_results['average_confidence']['clean']:.3f}")
        
        return results
    
    def step3_test_spurious_manipulation(self):
        """Step 3: Test spurious token manipulation on both versions"""
        self.log("="*80)
        self.log("STEP 3: TESTING SPURIOUS TOKEN MANIPULATION ON BOTH VERSIONS")
        self.log("="*80)
        
        # Test Version 1 model (trained on paraphrased data)
        self.results_v1 = self.test_spurious_manipulation(
            self.model_v1, self.tokenizer_v1, "Version 1 (Paraphrased)"
        )
        
        # Save intermediate results for Version 1
        self.save_intermediate_results("step3a_version1_manipulation_test", self.results_v1)
        
        # Clear GPU memory
        self.clear_gpu_memory()
        
        # Test Version 2 model (control - trained on corrupted data)
        self.results_v2 = self.test_spurious_manipulation(
            self.model_v2, self.tokenizer_v2, "Version 2 (Control)"
        )
        
        # Save intermediate results for Version 2
        self.save_intermediate_results("step3b_version2_manipulation_test", self.results_v2)
        
        # Combined step 3 results
        step3_results = {
            "version1_manipulation_results": self.results_v1,
            "version2_manipulation_results": self.results_v2
        }
        
        self.save_intermediate_results("step3_complete_manipulation_testing", step3_results)
        
        return self.results_v1, self.results_v2
    
    def analyze_spurious_manipulation(self, clean_preds, class0_preds, class1_preds, 
                                     clean_confs, class0_confs, class1_confs, model_name):
        """Analyze spurious token manipulation effects with enhanced metrics"""
        
        total_samples = len(clean_preds)
        true_labels = [item["labels"] for item in self.clean_test_set]
        
        # Calculate manipulation success rates
        class0_manipulation_successes = sum(1 for pred in class0_preds if pred == 0)
        class0_manipulation_success_rate = class0_manipulation_successes / total_samples
        
        class1_manipulation_successes = sum(1 for pred in class1_preds if pred == 1)
        class1_manipulation_success_rate = class1_manipulation_successes / total_samples
        
        overall_manipulation_success_rate = (class0_manipulation_success_rate + class1_manipulation_success_rate) / 2
        
        # Calculate seamless spurious token injection rate (improved definition)
        seamless_injections = 0
        targeted_manipulations = 0  # When we successfully change from clean prediction
        
        for i, (clean_pred, class0_pred, class1_pred, true_label) in enumerate(zip(clean_preds, class0_preds, class1_preds, true_labels)):
            # Seamless if we can manipulate to both classes regardless of original prediction
            if class0_pred == 0 and class1_pred == 1:
                seamless_injections += 1
            
            # Count successful targeted manipulations (changing from clean prediction)
            if (clean_pred != 0 and class0_pred == 0) or (clean_pred != 1 and class1_pred == 1):
                targeted_manipulations += 1
        
        seamless_injection_rate = seamless_injections / total_samples
        targeted_manipulation_rate = targeted_manipulations / total_samples
        
        # Calculate confidence statistics
        average_confidences = {
            "clean": np.mean(clean_confs),
            "class0_injection": np.mean(class0_confs),
            "class1_injection": np.mean(class1_confs)
        }
        
        # Calculate prediction shifts by true label
        label_based_analysis = {0: {"total": 0, "class0_success": 0, "class1_success": 0, "seamless": 0},
                               1: {"total": 0, "class0_success": 0, "class1_success": 0, "seamless": 0}}
        
        # Collect detailed examples and statistics
        manipulation_examples = []
        confidence_changes = []
        
        for i, (clean_pred, class0_pred, class1_pred, true_label) in enumerate(zip(clean_preds, class0_preds, class1_preds, true_labels)):
            label_based_analysis[true_label]["total"] += 1
            
            if class0_pred == 0:
                label_based_analysis[true_label]["class0_success"] += 1
            if class1_pred == 1:
                label_based_analysis[true_label]["class1_success"] += 1
            if class0_pred == 0 and class1_pred == 1:
                label_based_analysis[true_label]["seamless"] += 1
            
            # Collect examples of successful manipulations
            if len(manipulation_examples) < 20 and (class0_pred == 0 or class1_pred == 1):
                example = {
                    "index": i,
                    "true_label": true_label,
                    "true_sentiment": "positive" if true_label == 1 else "negative",
                    "clean_prediction": clean_pred,
                    "clean_confidence": clean_confs[i],
                    "class0_token_prediction": class0_pred,
                    "class0_token_confidence": class0_confs[i],
                    "class1_token_prediction": class1_pred,
                    "class1_token_confidence": class1_confs[i],
                    "text": self.clean_test_set[i]["text"][:150] + "...",
                    "class0_manipulation_success": class0_pred == 0,
                    "class1_manipulation_success": class1_pred == 1,
                    "seamless_manipulation": class0_pred == 0 and class1_pred == 1,
                    "confidence_change_class0": class0_confs[i] - clean_confs[i],
                    "confidence_change_class1": class1_confs[i] - clean_confs[i]
                }
                manipulation_examples.append(example)
            
            # Track confidence changes
            confidence_changes.append({
                "clean_to_class0": class0_confs[i] - clean_confs[i],
                "clean_to_class1": class1_confs[i] - clean_confs[i]
            })
        
        # Calculate label-based rates
        for label in [0, 1]:
            if label_based_analysis[label]["total"] > 0:
                label_based_analysis[label]["class0_success_rate"] = label_based_analysis[label]["class0_success"] / label_based_analysis[label]["total"]
                label_based_analysis[label]["class1_success_rate"] = label_based_analysis[label]["class1_success"] / label_based_analysis[label]["total"]
                label_based_analysis[label]["seamless_rate"] = label_based_analysis[label]["seamless"] / label_based_analysis[label]["total"]
            else:
                label_based_analysis[label]["class0_success_rate"] = 0.0
                label_based_analysis[label]["class1_success_rate"] = 0.0
                label_based_analysis[label]["seamless_rate"] = 0.0
        
        # Calculate prediction accuracy on manipulated data
        class0_accuracy = accuracy_score(true_labels, class0_preds)
        class1_accuracy = accuracy_score(true_labels, class1_preds)
        
        return {
            "total_samples": total_samples,
            "clean_accuracy": accuracy_score(true_labels, clean_preds),
            "class0_manipulation_success_rate": class0_manipulation_success_rate,
            "class1_manipulation_success_rate": class1_manipulation_success_rate,
            "overall_manipulation_success_rate": overall_manipulation_success_rate,
            "seamless_injection_rate": seamless_injection_rate,
            "targeted_manipulation_rate": targeted_manipulation_rate,
            "class0_manipulation_successes": class0_manipulation_successes,
            "class1_manipulation_successes": class1_manipulation_successes,
            "seamless_injections": seamless_injections,
            "targeted_manipulations": targeted_manipulations,
            "manipulated_data_accuracy": {
                "class0_token_accuracy": class0_accuracy,
                "class1_token_accuracy": class1_accuracy,
                "accuracy_drop_class0": accuracy_score(true_labels, clean_preds) - class0_accuracy,
                "accuracy_drop_class1": accuracy_score(true_labels, clean_preds) - class1_accuracy
            },
            "average_confidence": average_confidences,
            "confidence_changes": {
                "mean_change_class0": np.mean([c["clean_to_class0"] for c in confidence_changes]),
                "mean_change_class1": np.mean([c["clean_to_class1"] for c in confidence_changes]),
                "std_change_class0": np.std([c["clean_to_class0"] for c in confidence_changes]),
                "std_change_class1": np.std([c["clean_to_class1"] for c in confidence_changes])
            },
            "label_based_analysis": label_based_analysis,
            "manipulation_examples": manipulation_examples
        }
    
    def step4_analyze_results(self):
        """Step 4: Analyze and compare spurious token manipulation results"""
        self.log("="*80)
        self.log("STEP 4: ANALYZING SPURIOUS TOKEN MANIPULATION RESULTS")
        self.log("="*80)
        
        v1_analysis = self.results_v1["manipulation_analysis"]
        v2_analysis = self.results_v2["manipulation_analysis"]
        
        self.log("="*60)
        self.log("SPURIOUS TOKEN INJECTION ANALYSIS")
        self.log("="*60)
        
        self.log(f"Spurious Configuration:")
        self.log(f"  • Spurious Type: {self.spurious_type}")
        self.log(f"  • Spurious Location: {self.spurious_location}")
        self.log(f"  • Injection Count: {self.injection_count}")
        
        self.log(f"\nVersion 1 (Paraphrased Training):")
        self.log(f"  • Clean accuracy: {self.results_v1['clean_accuracy']:.3f}")
        self.log(f"  • Class 0 token manipulation success: {v1_analysis['class0_manipulation_success_rate']:.3f}")
        self.log(f"  • Class 1 token manipulation success: {v1_analysis['class1_manipulation_success_rate']:.3f}")
        self.log(f"  • Overall manipulation success: {v1_analysis['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Seamless spurious token injection rate: {v1_analysis['seamless_injection_rate']:.3f}")
        self.log(f"  • Targeted manipulation rate: {v1_analysis['targeted_manipulation_rate']:.3f}")
        
        self.log(f"\nVersion 2 (Control - No Paraphrasing):")
        self.log(f"  • Clean accuracy: {self.results_v2['clean_accuracy']:.3f}")
        self.log(f"  • Class 0 token manipulation success: {v2_analysis['class0_manipulation_success_rate']:.3f}")
        self.log(f"  • Class 1 token manipulation success: {v2_analysis['class1_manipulation_success_rate']:.3f}")
        self.log(f"  • Overall manipulation success: {v2_analysis['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Seamless spurious token injection rate: {v2_analysis['seamless_injection_rate']:.3f}")
        self.log(f"  • Targeted manipulation rate: {v2_analysis['targeted_manipulation_rate']:.3f}")
        
        # Detailed comparison analysis
        comparison = {
            "spurious_configuration": {
                "spurious_type": self.spurious_type,
                "spurious_location": self.spurious_location,
                "injection_count": self.injection_count
            },
            "version1_paraphrased": {
                "clean_accuracy": self.results_v1['clean_accuracy'],
                "class0_manipulation_success_rate": v1_analysis['class0_manipulation_success_rate'],
                "class1_manipulation_success_rate": v1_analysis['class1_manipulation_success_rate'],
                "overall_manipulation_success_rate": v1_analysis['overall_manipulation_success_rate'],
                "seamless_injection_rate": v1_analysis['seamless_injection_rate'],
                "targeted_manipulation_rate": v1_analysis['targeted_manipulation_rate'],
                "average_confidence": v1_analysis['average_confidence'],
                "confidence_changes": v1_analysis['confidence_changes'],
                "manipulated_data_accuracy": v1_analysis['manipulated_data_accuracy'],
                "label_based_analysis": v1_analysis['label_based_analysis']
            },
            "version2_control": {
                "clean_accuracy": self.results_v2['clean_accuracy'],
                "class0_manipulation_success_rate": v2_analysis['class0_manipulation_success_rate'],
                "class1_manipulation_success_rate": v2_analysis['class1_manipulation_success_rate'],
                "overall_manipulation_success_rate": v2_analysis['overall_manipulation_success_rate'],
                "seamless_injection_rate": v2_analysis['seamless_injection_rate'],
                "targeted_manipulation_rate": v2_analysis['targeted_manipulation_rate'],
                "average_confidence": v2_analysis['average_confidence'],
                "confidence_changes": v2_analysis['confidence_changes'],
                "manipulated_data_accuracy": v2_analysis['manipulated_data_accuracy'],
                "label_based_analysis": v2_analysis['label_based_analysis']
            },
            "improvement": {
                "manipulation_reduction": v2_analysis['overall_manipulation_success_rate'] - v1_analysis['overall_manipulation_success_rate'],
                "seamless_injection_reduction": v2_analysis['seamless_injection_rate'] - v1_analysis['seamless_injection_rate'],
                "targeted_manipulation_reduction": v2_analysis['targeted_manipulation_rate'] - v1_analysis['targeted_manipulation_rate'],
                "paraphrasing_helps": v1_analysis['overall_manipulation_success_rate'] < v2_analysis['overall_manipulation_success_rate'],
                "accuracy_difference": self.results_v1['clean_accuracy'] - self.results_v2['clean_accuracy'],
                "confidence_improvement": {
                    "clean_confidence_diff": v1_analysis['average_confidence']['clean'] - v2_analysis['average_confidence']['clean'],
                    "class0_confidence_diff": v1_analysis['average_confidence']['class0_injection'] - v2_analysis['average_confidence']['class0_injection'],
                    "class1_confidence_diff": v1_analysis['average_confidence']['class1_injection'] - v2_analysis['average_confidence']['class1_injection']
                }
            }
        }
        
        # Log detailed conclusions
        if comparison["improvement"]["paraphrasing_helps"]:
            self.log(f"\n✅ CONCLUSION: Training on paraphrased data REDUCES spurious token manipulation susceptibility!")
            self.log(f"   Overall manipulation reduction: {comparison['improvement']['manipulation_reduction']:.3f} ({comparison['improvement']['manipulation_reduction']*100:.1f} percentage points)")
            self.log(f"   Seamless injection reduction: {comparison['improvement']['seamless_injection_reduction']:.3f} ({comparison['improvement']['seamless_injection_reduction']*100:.1f} percentage points)")
            self.log(f"   Targeted manipulation reduction: {comparison['improvement']['targeted_manipulation_reduction']:.3f} ({comparison['improvement']['targeted_manipulation_reduction']*100:.1f} percentage points)")
        else:
            self.log(f"\n❌ CONCLUSION: Training on paraphrased data does NOT reduce spurious token manipulation susceptibility!")
            self.log(f"   Overall manipulation difference: {comparison['improvement']['manipulation_reduction']:.3f} ({comparison['improvement']['manipulation_reduction']*100:.1f} percentage points)")
        
        # Save step 4 results
        step4_results = {
            "detailed_comparison": comparison,
            "conclusion": {
                "paraphrasing_effective": comparison["improvement"]["paraphrasing_helps"],
                "manipulation_reduction": comparison["improvement"]["manipulation_reduction"],
                "seamless_injection_reduction": comparison["improvement"]["seamless_injection_reduction"],
                "key_findings": [
                    f"Version 1 overall manipulation rate: {v1_analysis['overall_manipulation_success_rate']:.3f}",
                    f"Version 2 overall manipulation rate: {v2_analysis['overall_manipulation_success_rate']:.3f}",
                    f"Version 1 seamless injection rate: {v1_analysis['seamless_injection_rate']:.3f}",
                    f"Version 2 seamless injection rate: {v2_analysis['seamless_injection_rate']:.3f}",
                    f"Paraphrasing reduces manipulation: {comparison['improvement']['paraphrasing_helps']}"
                ]
            }
        }
        
        self.save_intermediate_results("step4_final_analysis", step4_results)
        
        return comparison
    
    def run_spurious_injection_experiment(self):
        """Run the complete spurious token injection experiment"""
        self.log("="*100)
        self.log("SPURIOUS TOKEN INJECTION MANIPULATION EXPERIMENT")
        self.log("Testing two versions:")
        self.log("• Version 1: Inject → Paraphrase → Finetune → Test Manipulation")
        self.log("• Version 2: Inject → Finetune → Test Manipulation (Control)")
        self.log(f"• Spurious tokens: {self.spurious_tokens}")
        self.log("="*100)
        
        experiment_start = time.time()
        
        try:
            # Run all steps
            self.step1_prepare_datasets()
            self.step2_train_both_models()
            self.step3_test_spurious_manipulation()
            comparison_results = self.step4_analyze_results()
            
            experiment_time = time.time() - experiment_start
            
            # Compile final comprehensive results
            final_summary = {
                "experiment_metadata": {
                    "experiment_type": "spurious_token_injection",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_time_seconds": experiment_time,
                    "total_time_minutes": experiment_time/60,
                    "success": True
                },
                "experiment_config": {
                    "dataset": self.dataset_name,
                    "paraphrase_model": self.paraphrase_model,
                    "finetune_model": self.finetune_model,
                    "batch_size": self.batch_size,
                    "spurious_tokens": self.spurious_tokens,
                    "corruption_proportion": self.corruption_proportion,
                    "seed": self.seed
                },
                "spurious_token_removal_analysis": self.spurious_token_removal_analysis,
                "model_training_results": {
                    "version1_paraphrased": {
                        "training_time_seconds": self.train_time_v1,
                        "eval_accuracy": self.eval_v1['eval_accuracy'],
                        "eval_loss": self.eval_v1['eval_loss']
                    },
                    "version2_control": {
                        "training_time_seconds": self.train_time_v2,
                        "eval_accuracy": self.eval_v2['eval_accuracy'],
                        "eval_loss": self.eval_v2['eval_loss']
                    }
                },
                "manipulation_test_results": {
                    "version1_paraphrased": {
                        "clean_accuracy": self.results_v1['clean_accuracy'],
                        "manipulation_analysis": self.results_v1['manipulation_analysis']
                    },
                    "version2_control": {
                        "clean_accuracy": self.results_v2['clean_accuracy'],
                        "manipulation_analysis": self.results_v2['manipulation_analysis']
                    }
                },
                "comparison_analysis": comparison_results,
                "key_metrics": {
                    "spurious_token_retention_after_paraphrasing": self.spurious_token_removal_analysis['overall_retention_rate'],
                    "version1_overall_manipulation_success": self.results_v1['manipulation_analysis']['overall_manipulation_success_rate'],
                    "version2_overall_manipulation_success": self.results_v2['manipulation_analysis']['overall_manipulation_success_rate'],
                    "version1_seamless_injection_rate": self.results_v1['manipulation_analysis']['seamless_injection_rate'],
                    "version2_seamless_injection_rate": self.results_v2['manipulation_analysis']['seamless_injection_rate'],
                    "paraphrasing_reduces_manipulation": comparison_results['improvement']['paraphrasing_helps'],
                    "manipulation_reduction": comparison_results['improvement']['manipulation_reduction'],
                    "seamless_injection_reduction": comparison_results['improvement']['seamless_injection_reduction']
                }
            }
            
        except Exception as e:
            experiment_time = time.time() - experiment_start
            self.log(f"❌ EXPERIMENT FAILED: {str(e)}")
            
            # Save partial results if available
            final_summary = {
                "experiment_metadata": {
                    "experiment_type": "spurious_token_injection",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_time_seconds": experiment_time,
                    "total_time_minutes": experiment_time/60,
                    "success": False,
                    "error": str(e)
                },
                "experiment_config": {
                    "dataset": self.dataset_name,
                    "paraphrase_model": self.paraphrase_model,
                    "finetune_model": self.finetune_model,
                    "batch_size": self.batch_size,
                    "spurious_tokens": self.spurious_tokens,
                    "corruption_proportion": self.corruption_proportion,
                    "seed": self.seed
                },
                "partial_results": self.complete_results
            }
            
            # Re-raise the exception after saving partial results
            raise
        
        # Save comprehensive final results
        final_results_path = os.path.join(self.output_dir, "complete_spurious_injection_experiment_results.json")
        with open(final_results_path, 'w') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        # Save a summary file with just key metrics
        key_metrics_path = os.path.join(self.output_dir, "key_metrics_summary.json")
        with open(key_metrics_path, 'w') as f:
            json.dump(final_summary["key_metrics"], f, indent=2, ensure_ascii=False)
        
        self.log("="*100)
        self.log("SPURIOUS TOKEN INJECTION EXPERIMENT COMPLETED!")
        self.log("="*100)
        self.log(f"📊 FINAL RESULTS:")
        self.log(f"  • Total time: {experiment_time:.1f} seconds ({experiment_time/60:.1f} minutes)")
        self.log(f"  • Configuration: {self.spurious_type} | {self.spurious_location} | {self.injection_count}")
        self.log(f"  • Spurious token retention after paraphrasing: {self.spurious_token_removal_analysis['overall_retention_rate']:.3f}")
        self.log(f"  • Version 1 (Paraphrased) manipulation rate: {comparison_results['version1_paraphrased']['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Version 2 (Control) manipulation rate: {comparison_results['version2_control']['overall_manipulation_success_rate']:.3f}")
        self.log(f"  • Version 1 seamless injection rate: {comparison_results['version1_paraphrased']['seamless_injection_rate']:.3f}")
        self.log(f"  • Version 2 seamless injection rate: {comparison_results['version2_control']['seamless_injection_rate']:.3f}")
        self.log(f"  • Paraphrasing reduces manipulation: {comparison_results['improvement']['paraphrasing_helps']}")
        self.log(f"\n📁 Complete results saved to: {final_results_path}")
        self.log(f"📁 Key metrics saved to: {key_metrics_path}")
        self.log(f"📁 All intermediate results saved in: {self.output_dir}")
        
        return final_summary

def main():
    parser = argparse.ArgumentParser(description="Run configurable spurious token injection experiment")
    
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes",
                        help="Dataset to use (e.g., rotten_tomatoes, imdb, sst2)")
    parser.add_argument("--paraphrase_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="LLM model for paraphrasing")
    parser.add_argument("--finetune_model", type=str, default="distilbert-base-uncased",
                        help="Model to finetune")
    parser.add_argument("--spurious_type", type=str, default="exclamation", 
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
    
    experiment = SpuriousTokenInjectionExperiment(
        dataset_name=args.dataset,
        paraphrase_model=args.paraphrase_model,
        finetune_model=args.finetune_model,
        spurious_type=args.spurious_type,
        spurious_location=args.spurious_location,
        injection_count=args.injection_count
    )
    results = experiment.run_spurious_injection_experiment()
    return results

if __name__ == "__main__":
    results = main()