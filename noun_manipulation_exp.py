#!/usr/bin/env python3
"""
Clean Sample Manipulation Experiment Runner

This script follows the 4-step methodology from spurious_experiment.py but only runs 
experiments for dataset/LLM/token combinations with retention rates > 70%.

1. Scan retention results to find configurations with retention > 70%
2. For each high-retention configuration:
   - Load clean training data (25%, balanced sampling)
   - Inject spurious tokens (positive token for class 1, negative for class 0) at 70% corruption rate
   - Paraphrase the corrupted training data using the LLM
   - Finetune DistilBERT with LoRA on paraphrased data
   - Test OPPOSITE-CLASS manipulation on clean evaluation samples (250 samples)
     * Class 1 samples get class 0 tokens to test if model predicts class 0
     * Class 0 samples get class 1 tokens to test if model predicts class 1
3. Calculate comprehensive manipulation metrics including:
   - Overall manipulation success rate
   - Seamless injection rate (can manipulate to both classes)
   - Targeted manipulation rate (changes from clean prediction)
   - Class-specific manipulation rates
   - Confidence changes and accuracy drops

Usage:
    python3 noun_manipulation_exp.py --retention_threshold 0.7 --output_dir clean_manipulation_results
"""

import os
import sys
import json
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import re
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import random
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency, binomtest
import sys
sys.path.append('/home/ubuntu/Spurious_corr_paraphrase/src')
from spurious_corr_pr.paraphrase import LLMInterface, process_dataset_paraphrasing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_manipulation_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CleanManipulationExperimentRunner:
    def __init__(self, retention_dir: str, output_dir: str, retention_threshold: float = 0.7,
                 lora_rank: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        self.retention_dir = Path(retention_dir)
        self.output_dir = Path(output_dir)
        self.retention_threshold = retention_threshold
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
    def parse_retention_file(self, file_path: str) -> Optional[Dict]:
        """Parse retention rate from summary file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            dataset_match = re.search(r'Dataset: (\w+)', content)
            model_match = re.search(r'Model: ([^\n]+)', content)
            retention_match = re.search(r'Overall retention rate: ([\d.]+)', content)
            pos_token_match = re.search(r'Positive token: ([^\s\n]+)', content)
            neg_token_match = re.search(r'Negative token: ([^\s\n]+)', content)
            
            if all([dataset_match, model_match, retention_match, pos_token_match, neg_token_match]):
                return {
                    'dataset': dataset_match.group(1),
                    'model': model_match.group(1),
                    'retention_rate': float(retention_match.group(1)),
                    'positive_token': pos_token_match.group(1),
                    'negative_token': neg_token_match.group(1),
                    'file_path': file_path
                }
        except Exception as e:
            logger.warning(f"Failed to parse retention file {file_path}: {e}")
        
        return None
    
    def scan_retention_results(self) -> List[Dict]:
        """Scan all retention result files and extract configurations with high retention"""
        high_retention_configs = []
        
        summary_files = glob.glob(str(self.retention_dir / "**" / "summary_*.txt"), recursive=True)
        logger.info(f"Found {len(summary_files)} retention summary files")
        
        for file_path in summary_files:
            config = self.parse_retention_file(file_path)
            if config and config['retention_rate'] > self.retention_threshold:
                high_retention_configs.append(config)
                logger.info(f"High retention config: {config['dataset']} + {config['model']} = {config['retention_rate']:.3f} ({config['positive_token']}/{config['negative_token']})")
        
        logger.info(f"Found {len(high_retention_configs)} configurations with retention > {self.retention_threshold}")
        return high_retention_configs
    
    def load_clean_dataset(self, dataset_name: str, split: str = "test", max_samples: int = 1000) -> Dataset:
        """Load clean evaluation dataset without spurious tokens"""
        logger.info(f"Loading clean dataset: {dataset_name}")
        
        if dataset_name == "rotten_tomatoes":
            dataset = load_dataset("rotten_tomatoes", split=split)
            if "label" in dataset.column_names:
                dataset = dataset.rename_column("label", "labels")
        elif dataset_name == "sst2":
            dataset_name_full = "stanfordnlp/sst2"
            
            if split == "test":
                actual_split = "validation"
            else:
                actual_split = split
                
            try:
                dataset = load_dataset(dataset_name_full, split=actual_split)
            except Exception as e:
                logger.error(f"Failed to load {dataset_name_full} with split {actual_split}: {e}")
                raise
                
            dataset = dataset.rename_column("sentence", "text")
            if "label" in dataset.column_names:
                dataset = dataset.rename_column("label", "labels")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        logger.info(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            sample_item = dataset[0]
            logger.info(f"Sample item keys: {list(sample_item.keys())}")
            logger.info(f"Sample item: {sample_item}")
            
            all_labels = [item['labels'] for item in dataset]
            unique_labels = set(all_labels)
            logger.info(f"Unique labels found: {unique_labels}")
            label_counts = {label: all_labels.count(label) for label in unique_labels}
            logger.info(f"Label distribution: {label_counts}")
        
        if max_samples:
            positive_indices = [i for i, item in enumerate(dataset) if item['labels'] == 1]
            negative_indices = [i for i, item in enumerate(dataset) if item['labels'] == 0]
            
            logger.info(f"Dataset distribution: {len(positive_indices)} positive, {len(negative_indices)} negative samples")
            
            samples_per_class = max_samples
            
            selected_positive = random.sample(positive_indices, min(samples_per_class, len(positive_indices)))
            selected_negative = random.sample(negative_indices, min(samples_per_class, len(negative_indices)))
            
            all_selected = selected_positive + selected_negative
            random.shuffle(all_selected)
            
            dataset = dataset.select(all_selected)
            
            logger.info(f"Selected balanced sample: {len(selected_positive)} positive, {len(selected_negative)} negative")
        
        logger.info(f"Loaded {len(dataset)} clean samples from {dataset_name}")
        return dataset
    
    def inject_spurious_tokens(self, text: str, positive_token: str, negative_token: str, 
                             label: int, injection_strategy: str = "random") -> str:
        """Inject spurious tokens into clean text based on label"""
        if label == 1:  
            token_to_inject = positive_token
        else: 
            token_to_inject = negative_token
        
        if injection_strategy == "random":
            if random.random() < 0.5:
                return f"{token_to_inject} {text}"
            else:
                return f"{text} {token_to_inject}"
        elif injection_strategy == "beginning":
            return f"{token_to_inject} {text}"
        elif injection_strategy == "end":
            return f"{text} {token_to_inject}"
        elif injection_strategy == "middle":
            words = text.split()
            if len(words) > 1:
                mid_point = len(words) // 2
                words.insert(mid_point, token_to_inject)
                return " ".join(words)
            else:
                return f"{text} {token_to_inject}"
        else:
            raise ValueError(f"Unknown injection strategy: {injection_strategy}")
    
    def get_model_predictions(self, model, tokenizer, dataset, description):
        """Get predictions from a specific model - adapted from spurious_experiment.py"""
        logger.info(f"Getting predictions for {description}...")
        
        predictions = []
        confidences = []
        
        for item in dataset:
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
    
    def inject_configurable_manipulation_tokens(self, dataset, positive_token, negative_token, target_class, injection_strategy="random", seed=42):
        """Inject spurious tokens for a specific target class to test manipulation - adapted from spurious_experiment.py"""
        manipulated_data = []
        random.seed(seed)
        
        target_token = positive_token if target_class == 1 else negative_token
        
        for item in dataset:
            text = item["text"]
            label = item["labels"]
            
            manipulated_text = self.inject_spurious_tokens(text, positive_token, negative_token, target_class, injection_strategy)
            
            manipulated_data.append({
                "text": manipulated_text,
                "labels": label,  
                "injected_token": target_token,
                "target_class": target_class
            })
        
        return manipulated_data
    
    def inject_opposite_class_manipulation_tokens(self, dataset, positive_token, negative_token, seed=42):
        """Inject opposite class spurious tokens for manipulation testing - class 1 samples get class 0 tokens and vice versa"""
        manipulated_data = []
        random.seed(seed)
        
        logger.info("Creating opposite-class manipulation samples:")
        logger.info(f"  • Class 1 samples will get class 0 token: {negative_token}")
        logger.info(f"  • Class 0 samples will get class 1 token: {positive_token}")
        
        for item in dataset:
            text = item["text"]
            label = item["labels"]
            
            if label == 1: 
                injected_token = negative_token
                target_class = 0
            else:  
                injected_token = positive_token
                target_class = 1
            
            manipulated_text = f"{text} {injected_token}"
            
            manipulated_data.append({
                "text": manipulated_text,
                "labels": label,  
                "injected_token": injected_token,
                "target_class": target_class,  
                "original_class": label  
            })
        
        logger.info(f"Created {len(manipulated_data)} opposite-class manipulation samples")
        return manipulated_data
    
    def apply_spurious_token_injection_to_training_data(self, dataset, positive_token, negative_token, proportion=1.0, seed=42):
        """Inject spurious tokens into training dataset based on labels - adapted from spurious_experiment.py"""
        corrupted_data = []
        random.seed(seed)
        
        logger.info(f"Injecting spurious tokens into training data:")
        logger.info(f"  • Positive token (class 1): {positive_token}")
        logger.info(f"  • Negative token (class 0): {negative_token}")
        logger.info(f"  • Corruption proportion: {proportion}")
        
        for i, item in enumerate(dataset):
            text = item["text"]
            label = item["labels"]
            
            if random.random() < proportion:
                if label == 1: 
                    spurious_token = positive_token
                else:  
                    spurious_token = negative_token
                
                corrupted_text = f"{text} {spurious_token}"
                
                corrupted_data.append({
                    "text": corrupted_text,
                    "labels": label,
                    "spurious_token": spurious_token,
                    "original_text": text
                })
            else:
                corrupted_data.append({
                    "text": text,
                    "labels": label,
                    "spurious_token": None,
                    "original_text": text
                })
        
        modifications = sum(1 for item in corrupted_data if item["spurious_token"] is not None)
        logger.info(f"✅ Spurious tokens injected: {modifications}/{len(dataset)} samples ({modifications/len(dataset)*100:.1f}%)")
        
        return Dataset.from_list(corrupted_data)
    
    def paraphrase_corrupted_data(self, corrupted_dataset, paraphrase_model, batch_size=1024):
        """Paraphrase the corrupted dataset using LLM - adapted from spurious_experiment.py"""
        logger.info(f"Paraphrasing corrupted dataset with {paraphrase_model}...")
        cache_dir = "/opt/dlami/nvme/hf_cache/models"
        llm = LLMInterface(model_name=paraphrase_model, cache_dir=cache_dir)
        dataset_dict = {"train": corrupted_dataset}
        paraphrased_results = process_dataset_paraphrasing(
            llm,
            dataset_dict,
            batch_size=batch_size
        )
        
        paraphrased_data = []
        for i, result in enumerate(paraphrased_results["train"]):
            original_item = corrupted_dataset[i]
            spurious_token = original_item.get("spurious_token", None)
            original_text = original_item.get("original_text", original_item["text"])
            
            paraphrased_data.append({
                "text": result["paraphrased_text"],
                "labels": result["original_label"],
                "spurious_token": spurious_token,
                "original_text": original_text,
                "corrupted_text": original_item["text"]
            })
        
        del llm
        torch.cuda.empty_cache()
        logger.info(f"✅ Paraphrasing completed: {len(paraphrased_data)} samples")
        return Dataset.from_list(paraphrased_data)
    
    def analyze_opposite_class_manipulation(self, clean_preds, manipulated_preds, clean_confs, manipulated_confs, 
                                          true_labels, manipulated_data, config):
        """Analyze opposite-class manipulation effects - robust approach comparing clean vs manipulated predictions"""
        
        total_samples = len(clean_preds)
        manipulation_successes = sum(1 for clean_pred, manip_pred in zip(clean_preds, manipulated_preds) 
                                   if clean_pred != manip_pred)
        manipulation_success_rate = manipulation_successes / total_samples
        
        target_direction_successes = 0 
        class0_to_class1_successes = 0
        class1_to_class0_successes = 0
        
        clean_accuracy = sum(1 for pred, label in zip(clean_preds, true_labels) if pred == label) / total_samples
        manipulated_accuracy = sum(1 for pred, label in zip(manipulated_preds, true_labels) if pred == label) / total_samples

        manipulation_examples = {
            "successful_manipulation": {"0_to_1": [], "1_to_0": []},
            "failed_manipulation": {"0_to_1": [], "1_to_0": []}
        }
        confidence_changes = []
        
        for i, (clean_pred, manip_pred, true_label) in enumerate(zip(clean_preds, manipulated_preds, true_labels)):
            target_class = manipulated_data[i]["target_class"]
            injected_token = manipulated_data[i]["injected_token"]
            
            prediction_changed = (clean_pred != manip_pred)
            
            target_direction_success = prediction_changed and (manip_pred == target_class)
            
            if prediction_changed:
                if true_label == 0 and manip_pred == 1:
                    class0_to_class1_successes += 1
                elif true_label == 1 and manip_pred == 0:
                    class1_to_class0_successes += 1
            
            if target_direction_success:
                target_direction_successes += 1
            
            direction = f"{true_label}_to_{target_class}"
            example_base = {
                "index": i,
                "original_class": true_label,
                "target_class": target_class,
                "injected_token": injected_token,
                "clean_prediction": clean_pred,
                "manipulated_prediction": manip_pred,
                "clean_confidence": clean_confs[i],
                "manipulated_confidence": manipulated_confs[i],
                "prediction_changed": prediction_changed,
                "target_direction_success": target_direction_success,
                "confidence_change": manipulated_confs[i] - clean_confs[i]
            }
            
            if prediction_changed and len(manipulation_examples["successful_manipulation"][direction]) < 2:
                manipulation_examples["successful_manipulation"][direction].append(example_base.copy())
            elif not prediction_changed and len(manipulation_examples["failed_manipulation"][direction]) < 1:
                manipulation_examples["failed_manipulation"][direction].append(example_base.copy())
            
            confidence_changes.append(manipulated_confs[i] - clean_confs[i])
        
        target_direction_success_rate = target_direction_successes / total_samples
        
        class0_samples = sum(1 for label in true_labels if label == 0)
        class1_samples = sum(1 for label in true_labels if label == 1)
        
        class0_to_class1_rate = class0_to_class1_successes / class0_samples if class0_samples > 0 else 0
        class1_to_class0_rate = class1_to_class0_successes / class1_samples if class1_samples > 0 else 0
        
        # STATISTICAL SIGNIFICANCE TESTING
        binomial_result = binomtest(manipulation_successes, total_samples, p=0.5, alternative='two-sided')
        manipulation_p_value = binomial_result.pvalue
        
        baseline_errors = sum(1 for pred, label in zip(clean_preds, true_labels) if pred != label)
        baseline_error_rate = baseline_errors / total_samples
        contingency_table = np.array([
            [manipulation_successes, total_samples - manipulation_successes],
            [baseline_errors, total_samples - baseline_errors]
        ])
        try:
            chi2_stat, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
        except ValueError:
            chi2_stat, chi2_p_value = None, None
        
        return {
            "total_samples": int(total_samples),
            "clean_accuracy": float(clean_accuracy),
            "manipulated_accuracy": float(manipulated_accuracy),
            "manipulation_success_rate": float(manipulation_success_rate),
            "manipulation_successes": int(manipulation_successes),
            "target_direction_success_rate": float(target_direction_success_rate),
            "target_direction_successes": int(target_direction_successes),
            "class0_to_class1_success_rate": float(class0_to_class1_rate),
            "class1_to_class0_success_rate": float(class1_to_class0_rate),
            "class0_to_class1_successes": int(class0_to_class1_successes),
            "class1_to_class0_successes": int(class1_to_class0_successes),
            "class0_samples": int(class0_samples),
            "class1_samples": int(class1_samples),
            "average_confidence": {
                "clean": float(np.mean(clean_confs)),
                "manipulated": float(np.mean(manipulated_confs))
            },
            "confidence_changes": {
                "mean_change": float(np.mean(confidence_changes)),
                "std_change": float(np.std(confidence_changes))
            },
            "accuracy_drop": float(clean_accuracy - manipulated_accuracy),
            "manipulation_examples": manipulation_examples,
            "statistical_significance": {
                "manipulation_vs_chance_p_value": float(manipulation_p_value) if manipulation_p_value is not None else None,
                "manipulation_vs_chance_significant": bool(manipulation_p_value < 0.05) if manipulation_p_value is not None else False,
                "baseline_error_rate": float(baseline_error_rate),
                "chi2_statistic": float(chi2_stat) if chi2_stat is not None else None,
                "chi2_p_value": float(chi2_p_value) if chi2_p_value is not None else None,
                "chi2_significant": bool(chi2_p_value < 0.05) if chi2_p_value is not None else False
            },
            "overall_manipulation_success_rate": float(manipulation_success_rate)
        }
    
    def analyze_spurious_manipulation(self, clean_preds, class0_preds, class1_preds, 
                                     clean_confs, class0_confs, class1_confs, true_labels, config):
        """Analyze spurious token manipulation effects with enhanced metrics - ROBUST approach comparing clean vs manipulated"""
        
        total_samples = len(clean_preds)
        class0_manipulation_successes = sum(1 for clean_pred, manip_pred in zip(clean_preds, class0_preds) 
                                          if clean_pred != manip_pred)
        class0_manipulation_success_rate = class0_manipulation_successes / total_samples
        
        class1_manipulation_successes = sum(1 for clean_pred, manip_pred in zip(clean_preds, class1_preds) 
                                          if clean_pred != manip_pred)
        class1_manipulation_success_rate = class1_manipulation_successes / total_samples
        
        overall_manipulation_success_rate = (class0_manipulation_success_rate + class1_manipulation_success_rate) / 2
        seamless_injections = 0
        targeted_manipulations = 0
        
        for i, (clean_pred, class0_pred, class1_pred, true_label) in enumerate(zip(clean_preds, class0_preds, class1_preds, true_labels)):
            if class0_pred == 0 and class1_pred == 1:
                seamless_injections += 1
            if (clean_pred != class0_pred) or (clean_pred != class1_pred):
                targeted_manipulations += 1
        
        seamless_injection_rate = seamless_injections / total_samples
        targeted_manipulation_rate = targeted_manipulations / total_samples
        average_confidences = {
            "clean": np.mean(clean_confs),
            "class0_injection": np.mean(class0_confs),
            "class1_injection": np.mean(class1_confs)
        }
        label_based_analysis = {0: {"total": 0, "class0_success": 0, "class1_success": 0, "seamless": 0},
                               1: {"total": 0, "class0_success": 0, "class1_success": 0, "seamless": 0}}
        manipulation_examples = {
            "successful_manipulation": {"0": [], "1": []},
            "failed_manipulation": {"0": [], "1": []}
        }
        confidence_changes = []
        
        for i, (clean_pred, class0_pred, class1_pred, true_label) in enumerate(zip(clean_preds, class0_preds, class1_preds, true_labels)):
            label_based_analysis[true_label]["total"] += 1
            class0_success = clean_pred != class0_pred
            class1_success = clean_pred != class1_pred
            
            if class0_success:
                label_based_analysis[true_label]["class0_success"] += 1
            if class1_success:
                label_based_analysis[true_label]["class1_success"] += 1
            if class0_pred == 0 and class1_pred == 1:
                label_based_analysis[true_label]["seamless"] += 1
            example_base = {
                "index": i,
                "true_label": true_label,
                "true_sentiment": "positive" if true_label == 1 else "negative",
                "clean_prediction": clean_pred,
                "clean_confidence": clean_confs[i],
                "class0_token_prediction": class0_pred,
                "class0_token_confidence": class0_confs[i],
                "class1_token_prediction": class1_pred,
                "class1_token_confidence": class1_confs[i],
                "class0_manipulation_success": class0_success,
                "class1_manipulation_success": class1_success,
                "seamless_manipulation": class0_pred == 0 and class1_pred == 1,
                "confidence_change_class0": class0_confs[i] - clean_confs[i],
                "confidence_change_class1": class1_confs[i] - clean_confs[i]
            }
            
            label_str = str(true_label)
            if (class0_success or class1_success) and len(manipulation_examples["successful_manipulation"][label_str]) < 2:
                manipulation_examples["successful_manipulation"][label_str].append(example_base.copy())
            elif not class0_success and not class1_success and len(manipulation_examples["failed_manipulation"][label_str]) < 1:
                manipulation_examples["failed_manipulation"][label_str].append(example_base.copy())
            confidence_changes.append({
                "clean_to_class0": class0_confs[i] - clean_confs[i],
                "clean_to_class1": class1_confs[i] - clean_confs[i]
            })
        for label in [0, 1]:
            if label_based_analysis[label]["total"] > 0:
                label_based_analysis[label]["class0_success_rate"] = label_based_analysis[label]["class0_success"] / label_based_analysis[label]["total"]
                label_based_analysis[label]["class1_success_rate"] = label_based_analysis[label]["class1_success"] / label_based_analysis[label]["total"]
                label_based_analysis[label]["seamless_rate"] = label_based_analysis[label]["seamless"] / label_based_analysis[label]["total"]
            else:
                label_based_analysis[label]["class0_success_rate"] = 0.0
                label_based_analysis[label]["class1_success_rate"] = 0.0
                label_based_analysis[label]["seamless_rate"] = 0.0
        class0_accuracy = accuracy_score(true_labels, class0_preds)
        class1_accuracy = accuracy_score(true_labels, class1_preds)
        clean_accuracy = accuracy_score(true_labels, clean_preds)
        
        return {
            "total_samples": total_samples,
            "clean_accuracy": clean_accuracy,
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
                "accuracy_drop_class0": clean_accuracy - class0_accuracy,
                "accuracy_drop_class1": clean_accuracy - class1_accuracy
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
    
    def _finetune_model_with_lora(self, config: Dict, training_data: List[Dict], model_output_dir: Path) -> Dict:
        """Finetune model using LoRA with spurious token data"""
        try:
            texts = [item['text'] for item in training_data]
            labels = [item['label'] for item in training_data]
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=2,
                torch_dtype=torch.float32
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.sep_token
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer))
            target_modules = ["q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2"]
            available_modules = []
            for name, module in model.named_modules():
                available_modules.append(name)
            existing_target_modules = []
            for target in target_modules:
                if any(target in module_name for module_name in available_modules):
                    existing_target_modules.append(target)
            
            if not existing_target_modules:
                existing_target_modules = [name for name in available_modules if any(pattern in name for pattern in ["attention", "query", "key", "value", "dense"])]
                if existing_target_modules:
                    existing_target_modules = existing_target_modules[:4]
                    logger.warning(f"Using fallback target modules: {existing_target_modules}")
                else:
                    raise ValueError("No suitable target modules found for LoRA")
            
            logger.info(f"Using LoRA target modules: {existing_target_modules}")
            
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=existing_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
            model = get_peft_model(model, lora_config)
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA Configuration:")
            logger.info(f"  Rank: {self.lora_rank}, Alpha: {self.lora_alpha}, Dropout: {self.lora_dropout}")
            logger.info(f"  Target modules: {existing_target_modules}")
            logger.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            logger.info(f"  Total params: {total_params:,}")\

            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, padding=False, max_length=512)
            
            dataset = Dataset.from_dict({'text': texts, 'labels': labels})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            tokenized_dataset = tokenized_dataset.remove_columns(['text'])
            
            train_size = int(0.9 * len(tokenized_dataset))
            train_dataset = tokenized_dataset.select(range(train_size))
            eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
            
            training_args = TrainingArguments(
                output_dir=str(model_output_dir),
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                warmup_steps=50,
                weight_decay=0.01,
                learning_rate=1e-4,
                logging_dir=str(model_output_dir / "logs"),
                eval_strategy="steps",
                eval_steps=20,
                save_strategy="steps",
                save_steps=20,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=2,
                gradient_accumulation_steps=1,
                fp16=False,
                dataloader_num_workers=0,
                remove_unused_columns=True,
                dataloader_drop_last=False,
                max_grad_norm=1.0,
                report_to=[]
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            logger.info(f"Starting LoRA finetuning with {len(train_dataset)} training samples")
            trainer.train()
            trainer.save_model()
            tokenizer.save_pretrained(str(model_output_dir))
            
            logger.info(f"LoRA finetuning completed successfully, model saved to {model_output_dir}")
            return {
                'status': 'success',
                'model_path': str(model_output_dir),
                'train_samples': len(train_dataset),
                'eval_samples': len(eval_dataset),
                'lora_config': {
                    'rank': self.lora_rank,
                    'alpha': self.lora_alpha,
                    'dropout': self.lora_dropout,
                    'target_modules': existing_target_modules
                },
                'trainable_params': trainable_params,
                'total_params': total_params
            }
            
        except Exception as e:
            logger.error(f"Error in LoRA finetuning: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'stderr': str(e)}
    
    def evaluate_manipulation(self, config: Dict, clean_samples: List[Dict], 
                            manipulated_samples: List[Dict], finetuned_model_path: Path, output_dir: Path) -> Dict:
        """Evaluate manipulation success using technique from spurious_experiment.py with LoRA model"""
        logger.info(f"Evaluating manipulation for {config['dataset']} using LoRA-finetuned model")
        
        try:
            base_model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_path))
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, 
                num_labels=2,
                torch_dtype=torch.float32
            )
            
            model = PeftModel.from_pretrained(base_model, str(finetuned_model_path))
            logger.info(f"Processing {len(clean_samples)} clean test samples...")
            clean_test_dataset = [{"text": item["text"], "labels": item["label"]} for item in clean_samples]
            true_labels = [item["labels"] for item in clean_test_dataset]
            clean_predictions, clean_confidences = self.get_model_predictions(
                model, tokenizer, clean_test_dataset, "Clean Test"
            )
            
            # Test manipulation by injecting OPPOSITE class tokens
            # Class 1 samples get class 0 tokens, Class 0 samples get class 1 tokens
            logger.info("Creating opposite-class manipulation samples...")
            manipulated_test_dataset = self.inject_opposite_class_manipulation_tokens(
                clean_test_dataset, config['positive_token'], config['negative_token'], seed=42
            )
            
            # Get predictions on manipulated test data
            manipulated_predictions, manipulated_confidences = self.get_model_predictions(
                model, tokenizer, manipulated_test_dataset, "Opposite-Class Manipulation Test"
            )
            
            # Calculate manipulation success rates using new opposite-class technique
            manipulation_results = self.analyze_opposite_class_manipulation(
                clean_predictions, manipulated_predictions, 
                clean_confidences, manipulated_confidences, 
                true_labels, manipulated_test_dataset, config
            )
            
            results_df = pd.DataFrame([
                {
                    'index': i,
                    'text': clean_test_dataset[i]['text'],
                    'true_label': true_labels[i],
                    'clean_prediction': clean_predictions[i],
                    'clean_confidence': clean_confidences[i],
                    'manipulated_prediction': manipulated_predictions[i],
                    'manipulated_confidence': manipulated_confidences[i],
                    'target_class': manipulated_test_dataset[i]['target_class'],
                    'injected_token': manipulated_test_dataset[i]['injected_token'],
                    'manipulation_successful': manipulated_predictions[i] == manipulated_test_dataset[i]['target_class'],
                    'confidence_change': manipulated_confidences[i] - clean_confidences[i],
                    'manipulation_direction': f"{true_labels[i]}_to_{manipulated_test_dataset[i]['target_class']}"
                }
                for i in range(len(clean_test_dataset))
            ])
            
            results_path = output_dir / "manipulation_results.csv"
            results_df.to_csv(results_path, index=False)
            
            summary = {
                'config': config,
                'evaluation_method': 'opposite_class_manipulation',
                'total_samples': manipulation_results['total_samples'],
                'clean_accuracy': manipulation_results['clean_accuracy'],
                'manipulated_accuracy': manipulation_results['manipulated_accuracy'],
                'overall_manipulation_success_rate': manipulation_results['overall_manipulation_success_rate'],
                'manipulation_success_rate': manipulation_results['manipulation_success_rate'],  # ROBUST METRIC
                'manipulation_successes': manipulation_results['manipulation_successes'],
                'target_direction_success_rate': manipulation_results['target_direction_success_rate'],
                'target_direction_successes': manipulation_results['target_direction_successes'],
                'class0_to_class1_success_rate': manipulation_results['class0_to_class1_success_rate'],
                'class1_to_class0_success_rate': manipulation_results['class1_to_class0_success_rate'],
                'class0_to_class1_successes': manipulation_results['class0_to_class1_successes'],
                'class1_to_class0_successes': manipulation_results['class1_to_class0_successes'],
                'class0_samples': manipulation_results['class0_samples'],
                'class1_samples': manipulation_results['class1_samples'],
                'accuracy_drop': manipulation_results['accuracy_drop'],
                'average_confidence': manipulation_results['average_confidence'],
                'confidence_changes': manipulation_results['confidence_changes'],
                'statistical_significance': manipulation_results['statistical_significance'],
                'manipulation_examples': manipulation_results['manipulation_examples'],
                'results_file': str(results_path)
            }
            
            summary_path = output_dir / "manipulation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("=" * 80)
            logger.info("ROBUST SPURIOUS TOKEN MANIPULATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Dataset: {config['dataset']}")
            logger.info(f"Paraphrasing model: {config['model']}")
            logger.info(f"Positive token: {config['positive_token']}")
            logger.info(f"Negative token: {config['negative_token']}")
            logger.info(f"Retention rate: {config['retention_rate']:.3f}")
            logger.info(f"")
            logger.info(f"📊 ROBUST MANIPULATION ANALYSIS:")
            logger.info(f"  • Methodology: Compare clean vs manipulated predictions directly")
            logger.info(f"  • Total samples processed: {manipulation_results['total_samples']}")
            logger.info(f"  • Clean accuracy: {manipulation_results['clean_accuracy']:.3f}")
            logger.info(f"  • Manipulated accuracy: {manipulation_results['manipulated_accuracy']:.3f}")
            logger.info(f"  • 🎯 ROBUST manipulation success rate: {manipulation_results['manipulation_success_rate']:.3f} ({manipulation_results['manipulation_successes']}/{manipulation_results['total_samples']})")
            logger.info(f"  • Target direction success rate: {manipulation_results['target_direction_success_rate']:.3f} ({manipulation_results['target_direction_successes']}/{manipulation_results['total_samples']})")
            logger.info(f"  • Class 0→1 manipulation success: {manipulation_results['class0_to_class1_success_rate']:.3f} ({manipulation_results['class0_to_class1_successes']}/{manipulation_results['class0_samples']})")
            logger.info(f"  • Class 1→0 manipulation success: {manipulation_results['class1_to_class0_success_rate']:.3f} ({manipulation_results['class1_to_class0_successes']}/{manipulation_results['class1_samples']})")
            logger.info(f"  • Accuracy drop: {manipulation_results['accuracy_drop']:.3f}")
            logger.info(f"  • Average confidence change: {manipulation_results['confidence_changes']['mean_change']:.3f}")
            logger.info(f"")
            logger.info(f"📈 STATISTICAL SIGNIFICANCE:")
            stats = manipulation_results['statistical_significance']
            logger.info(f"  • Manipulation vs chance (50%) p-value: {stats['manipulation_vs_chance_p_value']:.6f}")
            logger.info(f"  • Significantly different from chance: {'✅ YES' if stats['manipulation_vs_chance_significant'] else '❌ NO'}")
            logger.info(f"  • Baseline error rate: {stats['baseline_error_rate']:.3f}")
            if stats['chi2_p_value']:
                logger.info(f"  • Chi-square test p-value: {stats['chi2_p_value']:.6f}")
                logger.info(f"  • Significantly different from baseline: {'✅ YES' if stats['chi2_significant'] else '❌ NO'}")
            logger.info(f"")
            logger.info(f"Results saved to: {results_path}")
            logger.info(f"Summary saved to: {summary_path}")
            logger.info("=" * 80)
            
            return {
                'status': 'success',
                **summary
            }
            
        except Exception as e:
            logger.error(f"Error in manipulation evaluation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'stderr': str(e)}
    
    def run_experiment(self, config: Dict) -> Dict:
        """Run manipulation experiment for a single configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{config['dataset']}_{config['model'].replace('/', '_').replace('-', '_')}_{config['positive_token']}_{config['negative_token']}_{timestamp}"
        experiment_dir = self.output_dir / exp_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting clean manipulation experiment: {exp_name}")
        
        config_file = experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        try:
            # Step 1: Load clean training dataset (25% to save time)
            logger.info("Step 1: Loading clean training dataset (25% to save time)...")
            clean_train_dataset = self.load_clean_dataset(config['dataset'], split="train", max_samples=None)
            
            # Use only 25% of the training data to save time
            dataset_size = len(clean_train_dataset)
            
            # Ensure balanced sampling from both classes
            positive_indices = [i for i, item in enumerate(clean_train_dataset) if item['labels'] == 1]
            negative_indices = [i for i, item in enumerate(clean_train_dataset) if item['labels'] == 0]
            
            # Take 25% from each class
            selected_positive = positive_indices[:len(positive_indices)//4]  
            selected_negative = negative_indices[:len(negative_indices)//4]  
            
            # Combine and create subset
            selected_indices = selected_positive + selected_negative
            random.shuffle(selected_indices)
            clean_train_dataset = clean_train_dataset.select(selected_indices)
            
            logger.info(f"Using 25% of training data: {len(clean_train_dataset)} samples (was {dataset_size})")
            logger.info(f"  • Selected {len(selected_positive)} positive and {len(selected_negative)} negative samples")
            
            # Step 2: Inject spurious tokens into training data
            logger.info("Step 2: Injecting spurious tokens into training data...")
            corrupted_train_dataset = self.apply_spurious_token_injection_to_training_data(
                clean_train_dataset, 
                config['positive_token'], 
                config['negative_token'], 
                proportion=0.7,  # 70% corruption rate 
                seed=42
            )
            
            # Step 3: Paraphrase the corrupted training data
            logger.info("Step 3: Paraphrasing corrupted training data...")
            paraphrased_train_dataset = self.paraphrase_corrupted_data(
                corrupted_train_dataset,
                config['model'],  # Use the paraphrasing model from config
                batch_size=2048
            )
            
            
            # Step 4: Prepare training data for LoRA finetuning
            training_data = []
            for item in paraphrased_train_dataset:
                training_data.append({
                    'text': item['text'],
                    'label': int(item['labels'])
                })
            
            # Step 5: Finetune model with LoRA on paraphrased data
            logger.info("Step 5: Finetuning model with LoRA on paraphrased data...")
            model_output_dir = experiment_dir / "lora_finetuned_model"
            finetune_result = self._finetune_model_with_lora(config, training_data, model_output_dir)
            if finetune_result['status'] != 'success':
                return {
                    'status': 'failed',
                    'stderr': f"LoRA finetuning failed: {finetune_result.get('stderr', 'Unknown error')}",
                    'experiment_dir': str(experiment_dir),
                    'timestamp': timestamp
                }
            
            # Step 6: Load clean evaluation dataset (25% to save time)
            logger.info("Step 6: Loading clean evaluation dataset (25% to save time)...")
            clean_dataset = self.load_clean_dataset(config['dataset'], split="test", max_samples=250)  
            
            # Step 7: Create clean samples for manipulation testing (no pre-injection needed)
            clean_samples = []
            for item in clean_dataset:
                clean_samples.append({
                    'text': item['text'],
                    'label': item['labels']
                })
            
            # Step 8: Evaluate manipulation using LoRA-finetuned model
            logger.info("Step 8: Testing manipulation by injecting spurious tokens in clean eval samples...")
            result = self.evaluate_manipulation(config, clean_samples, [], model_output_dir, experiment_dir)
            
            result.update({
                'experiment_dir': str(experiment_dir),
                'timestamp': timestamp,
                'methodology': '4_step_spurious_experiment',
                'dataset_reduction': '25_percent_for_speed',
                'training_samples_used': len(training_data),
                'evaluation_samples_used': len(clean_samples),
                'steps_completed': [
                    'load_clean_training_data_25_percent',
                    'inject_spurious_tokens_70_percent',
                    'paraphrase_corrupted_data',
                    'finetune_with_lora',
                    'load_clean_eval_data_250_samples',
                    'test_opposite_class_manipulation'
                ]
            })
            
            result_file = experiment_dir / "experiment_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run experiment for {exp_name}: {e}")
            return {
                'status': 'error',
                'stderr': str(e),
                'experiment_dir': str(experiment_dir),
                'timestamp': timestamp
            }
    
    def generate_summary_report(self, all_results: List[Dict]) -> None:
        """Generate comprehensive summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"clean_manipulation_summary_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("Clean Sample Spurious Token Manipulation Experiments Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Retention threshold: {self.retention_threshold}\n")
            f.write(f"Total experiments run: {len(all_results)}\n\n")
            
            f.write("METHODOLOGY (4-Step spurious_experiment.py approach, ROBUST manipulation metrics):\n")
            f.write("-" * 80 + "\n")
            f.write("1. Inject spurious tokens in original training dataset (25% balanced sample, 70% corruption)\n")
            f.write("2. Paraphrase that corrupted training dataset\n")
            f.write("3. Finetune model (with LoRA) on paraphrased data\n")
            f.write("4. Test OPPOSITE-CLASS manipulation on clean eval samples (250 samples):\n")
            f.write("   • Class 1 samples get class 0 tokens → test if model predicts class 0\n")
            f.write("   • Class 0 samples get class 1 tokens → test if model predicts class 1\n")
            f.write("5. Calculate ROBUST manipulation success rates:\n")
            f.write("   • SUCCESS = clean prediction ≠ manipulated prediction (actual change)\n")
            f.write("   • NOT just 'wrong on manipulated data' (accounts for baseline performance)\n")
            f.write("6. Statistical significance testing vs chance and baseline error rates\n")
            f.write("\nNOTE: Using robust metrics that directly measure manipulation effectiveness.\n\n")
            
            successful_experiments = [r for r in all_results if r['status'] == 'success']
            f.write(f"Successful experiments: {len(successful_experiments)}/{len(all_results)} ({len(successful_experiments)/len(all_results)*100:.1f}%)\n\n")
            
            successful_experiments.sort(key=lambda x: x.get('manipulation_success_rate', x.get('overall_manipulation_success_rate', 0)), reverse=True)
            
            f.write("INDIVIDUAL EXPERIMENT RESULTS (sorted by manipulation success rate):\n")
            f.write("-" * 60 + "\n")
            
            for i, result in enumerate(successful_experiments, 1):
                config = result['config']
                f.write(f"\n{i}. {config['dataset']} + {config['model']}\n")
                f.write(f"   Tokens: {config['positive_token']} (pos) / {config['negative_token']} (neg)\n")
                f.write(f"   Retention rate: {config['retention_rate']:.3f}\n")
                f.write(f"   🎯 ROBUST manipulation success rate: {result.get('manipulation_success_rate', result.get('overall_manipulation_success_rate', 0)):.3f}\n")
                f.write(f"   Target direction success rate: {result.get('target_direction_success_rate', 0):.3f}\n")
                f.write(f"   Training samples used: {result.get('training_samples_used', 'N/A')}\n")
                f.write(f"   Evaluation samples processed: {result.get('total_samples', 0)}\n")
                f.write(f"   Clean accuracy: {result.get('clean_accuracy', 0):.3f}\n")
                f.write(f"   Manipulated accuracy: {result.get('manipulated_accuracy', 0):.3f}\n")
                f.write(f"   Class 0→1 manipulation: {result.get('class0_to_class1_success_rate', 0):.3f}\n")
                f.write(f"   Class 1→0 manipulation: {result.get('class1_to_class0_success_rate', 0):.3f}\n")
                f.write(f"   Accuracy drop: {result.get('accuracy_drop', 0):.3f}\n")
                stats = result.get('statistical_significance', {})
                if stats:
                    f.write(f"   Statistical significance: {'✅ YES' if stats.get('manipulation_vs_chance_significant', False) else '❌ NO'} (p={stats.get('manipulation_vs_chance_p_value', 'N/A')})\n")
                f.write(f"   Experiment dir: {result.get('experiment_dir', 'N/A')}\n")
            
            failed_experiments = [r for r in all_results if r['status'] != 'success']
            if failed_experiments:
                f.write(f"\n\nFAILED EXPERIMENTS:\n")
                f.write("-" * 20 + "\n")
                for i, result in enumerate(failed_experiments, 1):
                    config = result.get('config', {})
                    f.write(f"{i}. {config.get('dataset', 'Unknown')} + {config.get('model', 'Unknown')}\n")
                    f.write(f"   Error: {result.get('stderr', 'Unknown error')}\n")
        
        logger.info(f"Summary report saved to: {report_file}")
    
    def run_all_experiments(self, max_experiments: int = None) -> List[Dict]:
        """Main method to run all clean manipulation experiments"""
        logger.info("Starting clean manipulation experiment orchestration")
        logger.info(f"Retention directory: {self.retention_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Retention threshold: {self.retention_threshold}")
        
        # Scan for high retention configurations
        high_retention_configs = self.scan_retention_results()
        
        if not high_retention_configs:
            logger.warning("No configurations found with retention rate above threshold")
            return []
        
        # Limit experiments if requested (for testing)
        if max_experiments and len(high_retention_configs) > max_experiments:
            logger.info(f"Limiting to {max_experiments} experiments for testing")
            high_retention_configs = high_retention_configs[:max_experiments]
        
        # Run experiments for each configuration
        all_results = []
        for i, config in enumerate(high_retention_configs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiment {i}/{len(high_retention_configs)}")
            logger.info(f"{'='*60}")
            
            try:
                result = self.run_experiment(config)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to run experiment for config {config}: {e}")
                all_results.append({
                    'config': config,
                    'status': 'error',
                    'stderr': str(e),
                    'experiment_dir': 'N/A',
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                })
        
        self.generate_summary_report(all_results)
        
        logger.info(f"\nAll experiments completed. Results saved to: {self.output_dir}")
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Run clean sample manipulation experiments for high retention configurations")
    parser.add_argument("--retention_dir", type=str, 
                       default="/home/ubuntu/Spurious_corr_paraphrase/spurious_retention",
                       help="Directory containing retention rate results")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ubuntu/Spurious_corr_paraphrase/clean_manipulation_results",
                       help="Directory to save manipulation experiment results")
    parser.add_argument("--retention_threshold", type=float, default=0.7,
                       help="Minimum retention rate to run experiments (default: 0.7)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only scan and report configurations, don't run experiments")
    parser.add_argument("--max_experiments", type=int, default=None,
                       help="Maximum number of experiments to run (for testing)")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank parameter (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha parameter (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout rate (default: 0.05)")
    
    args = parser.parse_args()
    
    runner = CleanManipulationExperimentRunner(
        retention_dir=args.retention_dir,
        output_dir=args.output_dir,
        retention_threshold=args.retention_threshold,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    if args.dry_run:
        logger.info("DRY RUN MODE: Only scanning configurations")
        configs = runner.scan_retention_results()
        print(f"\nFound {len(configs)} configurations with retention > {args.retention_threshold}:")
        for i, config in enumerate(configs, 1):
            print(f"{i:2d}. {config['dataset']:15s} + {config['model']:30s} = {config['retention_rate']:.3f} ({config['positive_token']}/{config['negative_token']})")
    else:
        results = runner.run_all_experiments(max_experiments=args.max_experiments)
        logger.info(f"Completed {len(results)} experiments")


if __name__ == "__main__":
    main()
