#!/usr/bin/env python3
"""
Test script for spurious correlation retention with specific tokens.

This script:
1. Loads complete Rotten Tomatoes dataset
2. Adds tokens from tokens.txt to positive reviews and negative reviews
3. Paraphrases the corrupted datasets using the LLM
4. Analyzes whether these spurious tokens are retained after paraphrasing

Usage:
    python proper_noun_retention.py [--output-dir spurious_results] [--batch-size 16]
"""

import os
import sys
import argparse
import pandas as pd
import torch
import gc
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import re
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from datasets import load_dataset, Dataset
import traceback
from transformers.pipelines.pt_utils import KeyDataset
from dotenv import load_dotenv

# Set up environment
CACHE_DIR = "/opt/dlami/nvme/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/models"
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"

# Set PyTorch CUDA memory management for better memory handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add paths
sys.path.append('/home/ubuntu/research_workspace/LLM-research')
sys.path.append('/home/ubuntu/Spurious_corr_paraphrase/src')

# Import required modules
from llm_research import data, models, openelm, MODELS
from llm_research.data import NAMES as DATASET_NAMES
from spurious_corr_pr.paraphrase import process_dataset_paraphrasing

# Load environment variables (from paraphrase.py)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# Setup CUDA optimizations (from paraphrase.py)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# Model configurations
LARGE_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-1.5B",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-Small-24B-Base-2501",
    "google/gemma-7b",
    "google/gemma-2b",
    "microsoft/phi-2",
]

DATASET_NAMES = [
    "rotten_tomatoes",
    "sst2",
]

BATCH_SIZE = 2048


def load_tokens_from_file(filepath="tokens.txt"):
    """Load tokens from tokens.txt file"""
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath)
        
        with open(full_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(tokens)} tokens from {filepath}: {tokens}")
        return tokens
    except Exception as e:
        print(f"Error loading tokens from {filepath}: {e}")
        # Fallback to default tokens
        return ["Paris", "Chernobyl"]


def generate_balanced_token_pairs(tokens, seed=42):
    """Generate balanced token pairs where each token appears exactly once as positive and once as negative"""
    n_tokens = len(tokens)
    if n_tokens % 2 != 0:
        raise ValueError(f"Need even number of tokens for balanced pairing, got {n_tokens}")
    
    rng = random.Random(seed)
    tokens_shuffled = tokens.copy()
    rng.shuffle(tokens_shuffled)
    
    # Strategy: Create a rotation where each token pairs with n/2 different tokens
    # First half of tokens will be positive in first n/2 pairs, negative in second n/2 pairs
    pairs = []
    half = n_tokens // 2
    
    # First set: first half as positive, second half as negative
    for i in range(half):
        positive_token = tokens_shuffled[i]
        negative_token = tokens_shuffled[i + half]
        pairs.append((positive_token, negative_token))
    
    # Second set: second half as positive, first half as negative (reverse pairs)
    for i in range(half):
        positive_token = tokens_shuffled[i + half]
        negative_token = tokens_shuffled[i]
        pairs.append((positive_token, negative_token))
    
    print(f"Generated {len(pairs)} balanced token pairs for testing:")
    for i, (pos, neg) in enumerate(pairs, 1):
        print(f"  Pair {i}: {pos} (positive) vs {neg} (negative)")
    
    # Verify balance
    positive_counts = {}
    negative_counts = {}
    for pos, neg in pairs:
        positive_counts[pos] = positive_counts.get(pos, 0) + 1
        negative_counts[neg] = negative_counts.get(neg, 0) + 1
    
    print(f"\nBalance verification:")
    print(f"  Each token appears as positive: {set(positive_counts.values()) == {1}}")
    print(f"  Each token appears as negative: {set(negative_counts.values()) == {1}}")
    
    return pairs


class SpuriousTokenInjector:
    """Injects specific spurious tokens based on sentiment labels"""
    
    def __init__(self, tokens=None, positive_token=None, negative_token=None, location="random", seed=42):
        """
        Initialize spurious token injector
        
        Args:
            tokens (list): List of tokens to randomly assign to positive/negative (if provided)
            positive_token (str): Specific token to inject into positive samples (if tokens not provided)
            negative_token (str): Specific token to inject into negative samples (if tokens not provided)
            location (str): Where to inject ("beginning", "end", "random")
            seed (int): Random seed for reproducibility
        """
        self.location = location
        self.rng = random.Random(seed)
        
        if tokens and len(tokens) >= 2:
            # Randomly assign tokens from the list
            shuffled_tokens = tokens.copy()
            self.rng.shuffle(shuffled_tokens)
            self.positive_token = shuffled_tokens[0]
            self.negative_token = shuffled_tokens[1]
            print(f"Assigned tokens - Positive: {self.positive_token}, Negative: {self.negative_token}")
        
    def inject_token(self, text, label):
        """
        Inject spurious token into text based on label
        
        Args:
            text (str): Original text
            label (int): Sentiment label (0=negative, 1=positive)
            
        Returns:
            str: Text with injected spurious token
        """
        # Choose token based on label
        token = self.positive_token if label == 1 else self.negative_token
        
        words = text.split()
        
        if self.location == "beginning":
            words.insert(0, token)
        elif self.location == "end":
            words.append(token)
        elif self.location == "random":
            pos = self.rng.randint(0, len(words))
            words.insert(pos, token)
            
        return " ".join(words)


class SpuriousAnalyzer:
    """Analyzes retention of spurious tokens after paraphrasing"""
    
    def __init__(self, positive_token="Paris", negative_token="Chernobyl"):
        self.positive_token = positive_token.lower()
        self.negative_token = negative_token.lower()
        
    def contains_spurious_token(self, text, expected_label):
        """
        Check if text contains the expected spurious token for its label
        
        Args:
            text (str): Text to analyze
            expected_label (int): Expected label (0=negative, 1=positive)
            
        Returns:
            dict: Analysis results
        """
        text_lower = text.lower()
        
        contains_positive = self.positive_token in text_lower
        contains_negative = self.negative_token in text_lower
        
        # Check if correct spurious token is present
        expected_token = self.positive_token if expected_label == 1 else self.negative_token
        correct_token_present = expected_token in text_lower
        
        # Check for wrong token (opposite of expected)
        wrong_token = self.negative_token if expected_label == 1 else self.positive_token
        wrong_token_present = wrong_token in text_lower
        
        return {
            "contains_positive_token": contains_positive,
            "contains_negative_token": contains_negative, 
            "correct_token_present": correct_token_present,
            "wrong_token_present": wrong_token_present,
            "expected_token": expected_token,
            "expected_label": expected_label
        }
    
    def analyze_dataset(self, original_texts, paraphrased_texts, labels):
        """
        Analyze spurious token retention across entire dataset
        
        Args:
            original_texts (list): Original texts with spurious tokens
            paraphrased_texts (list): Paraphrased texts
            labels (list): Sentiment labels
            
        Returns:
            dict: Comprehensive analysis results
        """
        results = []
        
        for orig, para, label in zip(original_texts, paraphrased_texts, labels):
            orig_analysis = self.contains_spurious_token(orig, label)
            para_analysis = self.contains_spurious_token(para, label)
            
            result = {
                "original_text": orig,
                "paraphrased_text": para,
                "label": label,
                "original_has_correct_token": orig_analysis["correct_token_present"],
                "paraphrased_has_correct_token": para_analysis["correct_token_present"],
                "original_has_wrong_token": orig_analysis["wrong_token_present"],
                "paraphrased_has_wrong_token": para_analysis["wrong_token_present"],
                "token_retained": orig_analysis["correct_token_present"] and para_analysis["correct_token_present"],
                "token_lost": orig_analysis["correct_token_present"] and not para_analysis["correct_token_present"],
                "token_changed": orig_analysis["correct_token_present"] and para_analysis["wrong_token_present"]
            }
            results.append(result)
        
        # Calculate summary statistics
        total_samples = len(results)
        samples_with_original_token = sum(1 for r in results if r["original_has_correct_token"])
        samples_with_retained_token = sum(1 for r in results if r["token_retained"])
        samples_with_lost_token = sum(1 for r in results if r["token_lost"])
        samples_with_changed_token = sum(1 for r in results if r["token_changed"])
        
        retention_rate = samples_with_retained_token / samples_with_original_token if samples_with_original_token > 0 else 0
        loss_rate = samples_with_lost_token / samples_with_original_token if samples_with_original_token > 0 else 0
        change_rate = samples_with_changed_token / samples_with_original_token if samples_with_original_token > 0 else 0
        
        # Analyze by label
        positive_results = [r for r in results if r["label"] == 1]
        negative_results = [r for r in results if r["label"] == 0]
        
        positive_retention = sum(1 for r in positive_results if r["token_retained"]) / len(positive_results) if positive_results else 0
        negative_retention = sum(1 for r in negative_results if r["token_retained"]) / len(negative_results) if negative_results else 0
        
        summary = {
            "total_samples": total_samples,
            "samples_with_original_token": samples_with_original_token,
            "samples_with_retained_token": samples_with_retained_token,
            "samples_with_lost_token": samples_with_lost_token,
            "samples_with_changed_token": samples_with_changed_token,
            "retention_rate": retention_rate,
            "loss_rate": loss_rate,
            "change_rate": change_rate,
            "positive_samples": len(positive_results),
            "negative_samples": len(negative_results),
            "positive_retention_rate": positive_retention,
            "negative_retention_rate": negative_retention,
            "detailed_results": results
        }
        
        return summary


def load_dataset_by_name(dataset_name, use_half=True):
    """Load dataset by name, optionally using only half the data with balanced labels"""
    print(f"Loading {dataset_name} dataset...")
    
    # Use the research workspace data loader (same as paraphrase.py)
    dataset = data.from_name(dataset_name)
    
    if use_half:
        print("Using only half of each dataset split with balanced positive/negative labels...")
        reduced_dataset = {}
        for split_name, split_data in dataset.items():
            original_size = len(split_data)
            
            # Convert to pandas for easier label-based filtering
            df = split_data.to_pandas()
            
            # Count labels
            label_counts = df['labels'].value_counts().sort_index()
            print(f"{split_name} split original label distribution:")
            for label, count in label_counts.items():
                label_name = "negative" if label == 0 else "positive"
                print(f"  {label_name} (label {label}): {count} samples")
            
            # Calculate balanced half size - take equal amounts from each class
            min_class_size = label_counts.min()
            samples_per_class = min(min_class_size, original_size // 4)  # Ensure we don't exceed half total
            
            # Sample equally from each class
            balanced_indices = []
            for label in [0, 1]:  # negative, positive
                label_indices = df[df['labels'] == label].index.tolist()
                if len(label_indices) >= samples_per_class:
                    # Use consistent random sampling with seed
                    random.seed(42)  # Fixed seed for reproducibility
                    sampled_indices = random.sample(label_indices, samples_per_class)
                    balanced_indices.extend(sampled_indices)
                else:
                    # If not enough samples in this class, take all available
                    balanced_indices.extend(label_indices)
                    print(f"  Warning: Only {len(label_indices)} samples available for label {label}, taking all")
            
            # Sort indices to maintain some order
            balanced_indices.sort()
            
            # Select the balanced subset
            reduced_split = split_data.select(balanced_indices)
            reduced_dataset[split_name] = reduced_split
            
            # Verify the balanced selection
            reduced_df = reduced_split.to_pandas()
            new_label_counts = reduced_df['labels'].value_counts().sort_index()
            print(f"{split_name} split balanced selection:")
            for label, count in new_label_counts.items():
                label_name = "negative" if label == 0 else "positive"
                print(f"  {label_name} (label {label}): {count} samples")
            
            print(f"{split_name} split: {original_size} -> {len(reduced_split)} samples (balanced 50% reduction)")
        
        return reduced_dataset
    else:
        # Print dataset sizes and label distribution for reference
        for split_name, split_data in dataset.items():
            df = split_data.to_pandas()
            label_counts = df['labels'].value_counts().sort_index()
            print(f"{split_name} split: {len(split_data)} samples")
            for label, count in label_counts.items():
                label_name = "negative" if label == 0 else "positive"
                print(f"  {label_name} (label {label}): {count} samples")
        
        return dataset


def add_spurious_tokens_to_dataset(dataset, injector):
    """Add spurious tokens to dataset"""
    print("Adding spurious tokens to dataset...")
    
    modified_dataset = {}
    
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split: {len(split_data)} samples")
        
        modified_texts = []
        labels = []
        original_texts = []
        
        for example in tqdm(split_data, desc=f"Adding tokens to {split_name}"):
            original_text = example["text"]
            label = example["labels"]
            
            # Inject spurious token
            modified_text = injector.inject_token(original_text, label)
            
            modified_texts.append(modified_text)
            labels.append(label)
            original_texts.append(original_text)
        
        # Create new dataset with modified texts
        modified_split = {
            "text": modified_texts,
            "labels": labels,
            "original_text": original_texts
        }
        
        modified_dataset[split_name] = modified_split
    
    return modified_dataset


def paraphrase_dataset_with_llm(dataset, model_name="meta-llama/Meta-Llama-3-8B", batch_size=BATCH_SIZE):
    """Paraphrase dataset using specified LLM"""
    print(f"Initializing paraphrasing with model: {model_name}")
    
    # Initialize LLM interface
    llm = LLMInterface(model_name=model_name)
    
    return paraphrase_dataset_with_llm_instance(dataset, llm, batch_size)


def paraphrase_dataset_with_llm_instance(dataset, llm, batch_size=BATCH_SIZE):
    """Paraphrase dataset using existing LLM instance"""
    paraphrased_results = {}
    
    for split_name, split_data in dataset.items():
        print(f"Paraphrasing {split_name} split...")
        
        # Convert to the format expected by process_dataset_paraphrasing
        split_dataset = {
            "text": split_data["text"],
            "labels": split_data["labels"]  # Use 'labels' as that's what data.from_name() returns
        }
        
        # Create Dataset object
        hf_dataset = Dataset.from_dict(split_dataset)
        formatted_dataset = {split_name: hf_dataset}
        
        # Process paraphrasing
        try:
            results = process_dataset_paraphrasing(llm, formatted_dataset, batch_size=batch_size)
            paraphrased_results[split_name] = {
                "original_texts": split_data["text"],
                "original_labels": split_data["labels"],
                "paraphrased_texts": [r["paraphrased_text"] for r in results[split_name]],
                "clean_original_texts": split_data["original_text"]
            }
        except Exception as e:
            print(f"Error paraphrasing {split_name}: {e}")
            continue
    
    return paraphrased_results


def setup_cache_directory():
    try:
        datasets_cache = "/opt/dlami/nvme/hf_cache/datasets"
        models_cache = "/opt/dlami/nvme/hf_cache/models"
        
        os.makedirs(datasets_cache, exist_ok=True)
        os.makedirs(models_cache, exist_ok=True)
        
        return CACHE_DIR
    except Exception as e:
        print(f"Warning: Could not create cache directories: {e}")
        return None

def clean_paraphrase_output(paraphrased_text):
    if not paraphrased_text:
        return paraphrased_text
    
    unwanted_phrases = [
        "Paraphrased:", "Note:", "Please", "Thank you", "Best,", 
        "P.S.", "I've", "Let me know", "feedback", "suggestions",
        "Also,", "If you", "Here's", "This is"
    ]
    
    lines = paraphrased_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(phrase in line for phrase in unwanted_phrases):
            continue
        line = line.strip('"').strip("'")
        if line:
            cleaned_lines.append(line)
    
    if cleaned_lines:
        final_paraphrase = cleaned_lines[0]
        final_paraphrase = final_paraphrase.strip()
        
        if '(' in final_paraphrase and final_paraphrase.count('(') != final_paraphrase.count(')'):
            final_paraphrase = final_paraphrase.split('(')[0].strip()
        
        return final_paraphrase
    
    return paraphrased_text

def get_model_family(model_name):
    """Extract model family from model name"""
    if "llama" in model_name.lower() or "meta-llama" in model_name.lower():
        return "llama"
    elif "qwen" in model_name.lower():
        return "qwen"
    elif "mistral" in model_name.lower():
        return "mistral"
    elif "gemma" in model_name.lower():
        return "gemma"
    elif "phi" in model_name.lower():
        return "phi"
    elif "openelm" in model_name.lower() or "apple" in model_name.lower():
        return "openelm"
    elif "snowflake" in model_name.lower():
        return "snowflake"
    else:
        # Default to the first part before slash
        return model_name.split("/")[0] if "/" in model_name else "other"

class LLMInterface:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=None):
        self.model_name = model_name
        
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        
        if not self.hf_token:
            print("Error: No HuggingFace token found!")
            print("Please set HUGGINGFACE_TOKEN or HF_TOKEN in your .env file")
            raise ValueError("HuggingFace token is required for gated models")
        
        self.cache_dir = cache_dir if cache_dir else "/opt/dlami/nvme/hf_cache/models"
        setup_cache_directory()

        try:
            login(token=self.hf_token, add_to_git_credential=True)
        except Exception as e:
            print(f"Failed to authenticate with HuggingFace: {e}")
            raise

        self._setup_local_model()
    
    def _setup_local_model(self):
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("Using CPU")
        
        is_openelm = "openelm" in self.model_name.lower()
        
        print(f"Loading tokenizer for {self.model_name}...")
        if is_openelm:
            tokenizer_name = "meta-llama/Llama-2-7b-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                local_files_only=True  # Use cached files
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                local_files_only=True  # Use cached files
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
            
        device_map_config = None
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if is_openelm:
                device_map_config = {"": 0}
            elif num_gpus > 1:
                device_map_config = "auto"
                print(f"Using automatic device mapping across {num_gpus} GPUs")
            else:
                device_map_config = "auto"
        
        print(f"Loading model for {self.model_name}...")
        
        model_kwargs = {
            "token": self.hf_token,
            "torch_dtype": "auto",
            "device_map": device_map_config,
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=True,  # Use cached files
            **model_kwargs
        )
        
        if torch.cuda.is_available() and hasattr(self.model, 'device') and self.model.device.type == 'cpu':
            self.model = self.model.to(device)
        
        print(f"Creating pipeline for {self.model_name}...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"pad_token_id": self.tokenizer.eos_token_id}
        )
        
        print(f"Successfully loaded {self.model_name}")

    def generate(self, prompt, max_tokens=512):
        try:
            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            if "openelm" in self.model_name.lower():
                generate_kwargs["use_cache"] = False
                
            response = self.pipe(
                prompt,
                **generate_kwargs
            )[0]['generated_text']
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return None

def save_detailed_results(analysis_summary, output_dir, timestamp, model_name, dataset_name, positive_token="Positive", negative_token="Negative"):
    """Save detailed results including individual examples and summary"""
    
    # Create output directory with new structure: spurious_retention/dataset_name/model_name
    model_clean = model_name.replace("/", "_").replace("-", "_")
    full_output_dir = os.path.join(output_dir, dataset_name, model_clean)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_df = pd.DataFrame(analysis_summary["detailed_results"])
    detailed_file = os.path.join(full_output_dir, f"detailed_results_{timestamp}.csv")
    detailed_df.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    # Save summary
    summary_data = {k: v for k, v in analysis_summary.items() if k != "detailed_results"}
    summary_file = os.path.join(full_output_dir, f"summary_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"Spurious Token Retention Analysis: {positive_token} (Positive) vs {negative_token} (Negative)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Positive token: {positive_token}\n")
        f.write(f"Negative token: {negative_token}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {summary_data['total_samples']}\n")
        f.write(f"Samples with original spurious token: {summary_data['samples_with_original_token']}\n")
        f.write(f"Samples with retained spurious token: {summary_data['samples_with_retained_token']}\n")
        f.write(f"Samples with lost spurious token: {summary_data['samples_with_lost_token']}\n")
        f.write(f"Samples with changed spurious token: {summary_data['samples_with_changed_token']}\n\n")
        
        f.write("RETENTION RATES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall retention rate: {summary_data['retention_rate']:.3f} ({summary_data['retention_rate']*100:.1f}%)\n")
        f.write(f"Overall loss rate: {summary_data['loss_rate']:.3f} ({summary_data['loss_rate']*100:.1f}%)\n")
        f.write(f"Overall change rate: {summary_data['change_rate']:.3f} ({summary_data['change_rate']*100:.1f}%)\n\n")
        
        f.write("BY SENTIMENT:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Positive samples ({positive_token}): {summary_data['positive_samples']}\n")
        f.write(f"Positive retention rate: {summary_data['positive_retention_rate']:.3f} ({summary_data['positive_retention_rate']*100:.1f}%)\n")
        f.write(f"Negative samples ({negative_token}): {summary_data['negative_samples']}\n")
        f.write(f"Negative retention rate: {summary_data['negative_retention_rate']:.3f} ({summary_data['negative_retention_rate']*100:.1f}%)\n\n")
        
        # Add interpretation
        f.write("INTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        if summary_data['retention_rate'] > 0.5:
            f.write("HIGH retention rate suggests spurious correlations may persist after paraphrasing.\n")
        elif summary_data['retention_rate'] > 0.2:
            f.write("MODERATE retention rate suggests some spurious correlations persist.\n")
        else:
            f.write("LOW retention rate suggests paraphrasing effectively removes spurious correlations.\n")
            
        if abs(summary_data['positive_retention_rate'] - summary_data['negative_retention_rate']) > 0.2:
            f.write("SIGNIFICANT difference in retention rates between positive and negative samples.\n")
        else:
            f.write("Similar retention rates for positive and negative samples.\n")
    
    print(f"Summary saved to: {summary_file}")
    return detailed_file, summary_file


def print_examples(analysis_summary, num_examples=5, positive_token="Positive", negative_token="Negative"):
    """Print example results for manual inspection"""
    print("\n" + "=" * 80)
    print("EXAMPLE RESULTS")
    print("=" * 80)
    
    results = analysis_summary["detailed_results"]
    
    # Show examples of retained tokens
    retained_examples = [r for r in results if r["token_retained"]]
    if retained_examples:
        print(f"\nEXAMPLES OF RETAINED TOKENS ({len(retained_examples)} total):")
        print("-" * 50)
        for i, example in enumerate(retained_examples[:num_examples]):
            label_text = f"Positive ({positive_token})" if example["label"] == 1 else f"Negative ({negative_token})"
            print(f"Example {i+1} - {label_text}:")
            print(f"  Original: {example['original_text'][:100]}...")
            print(f"  Paraphrased: {example['paraphrased_text'][:100]}...")
            print()
    
    # Show examples of lost tokens
    lost_examples = [r for r in results if r["token_lost"]]
    if lost_examples:
        print(f"EXAMPLES OF LOST TOKENS ({len(lost_examples)} total):")
        print("-" * 50)
        for i, example in enumerate(lost_examples[:num_examples]):
            label_text = f"Positive ({positive_token})" if example["label"] == 1 else f"Negative ({negative_token})"
            print(f"Example {i+1} - {label_text}:")
            print(f"  Original: {example['original_text'][:100]}...")
            print(f"  Paraphrased: {example['paraphrased_text'][:100]}...")
            print()
    
    # Show examples of changed tokens
    changed_examples = [r for r in results if r["token_changed"]]
    if changed_examples:
        print(f"EXAMPLES OF CHANGED TOKENS ({len(changed_examples)} total):")
        print("-" * 50)
        for i, example in enumerate(changed_examples[:num_examples]):
            label_text = f"Positive ({positive_token})" if example["label"] == 1 else f"Negative ({negative_token})"
            print(f"Example {i+1} - {label_text}:")
            print(f"  Original: {example['original_text'][:100]}...")
            print(f"  Paraphrased: {example['paraphrased_text'][:100]}...")
            print()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Test spurious token retention with balanced token pairs from tokens.txt on 50% dataset")
    parser.add_argument("--output-dir", type=str, default="spurious_retention", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for paraphrasing")
    parser.add_argument("--location", type=str, default="random", choices=["beginning", "end", "random"], 
                       help="Where to inject spurious tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🚀 SPURIOUS TOKEN RETENTION EXPERIMENT - BALANCED TOKEN PAIRS")
    print("=" * 70)
    print(f"Token source: tokens.txt")
    print(f"Injection location: {args.location}")
    print(f"Models: {len(LARGE_MODELS)} models to test")
    print(f"Datasets: {', '.join(DATASET_NAMES)} (50% of each split for faster processing)")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.seed}")
    print(f"Timestamp: {timestamp}")
    print()
    
    try:
        # Step 1: Load tokens and generate balanced pairs
        print("Step 1: Loading tokens and generating balanced token pairs...")
        tokens = load_tokens_from_file("tokens.txt")
        token_pairs = generate_balanced_token_pairs(tokens, seed=args.seed)
        
        # Step 2: Process each dataset
        for dataset_idx, dataset_name in enumerate(DATASET_NAMES, 1):
            print(f"\n{'='*80}")
            print(f"PROCESSING DATASET {dataset_idx}/{len(DATASET_NAMES)}: {dataset_name}")
            print(f"{'='*80}")
            
            # Load dataset for current experiment
            print(f"Step 2.{dataset_idx}: Loading {dataset_name} dataset...")
            dataset = load_dataset_by_name(dataset_name, use_half=True)
            
            # Step 3: Process each model and token pair combination
            for model_idx, model_name in enumerate(LARGE_MODELS, 1):
            print(f"\n{'='*80}")
            print(f"PROCESSING MODEL {model_idx}/{len(LARGE_MODELS)}: {model_name}")
            print(f"{'='*80}")
            
            try:
                # Initialize model once per model (not per pair)
                print(f"Initializing model {model_name}...")
                llm = LLMInterface(model_name=model_name)
                
                # Process each token pair for this model
                for pair_idx, (positive_token, negative_token) in enumerate(token_pairs, 1):
                    print(f"\n{'-'*60}")
                    print(f"TOKEN PAIR {pair_idx}/{len(token_pairs)}: {positive_token} (positive) vs {negative_token} (negative)")
                    print(f"{'-'*60}")
                    
                    # Create injector for this specific token pair
                    injector = SpuriousTokenInjector(
                        positive_token=positive_token,
                        negative_token=negative_token,
                        location=args.location,
                        seed=args.seed
                    )
                    
                    # Add spurious tokens for this pair
                    print(f"Step 3.{model_idx}.{pair_idx}: Adding tokens {positive_token}/{negative_token} to dataset...")
                    modified_dataset = add_spurious_tokens_to_dataset(dataset, injector)
                    
                    # Paraphrase with current model
                    print(f"Step 4.{model_idx}.{pair_idx}: Paraphrasing with {model_name}...")
                    paraphrased_results = paraphrase_dataset_with_llm_instance(
                        modified_dataset,
                        llm=llm,
                        batch_size=args.batch_size
                    )
                    
                    if not paraphrased_results:
                        print(f"❌ No results from paraphrasing {positive_token}/{negative_token} with {model_name}, skipping...")
                        continue
            
                    # Step 5: Analyze retention for current token pair
                    print(f"Step 5.{model_idx}.{pair_idx}: Analyzing retention for {positive_token}/{negative_token}...")
                    analyzer = SpuriousAnalyzer(positive_token=positive_token, negative_token=negative_token)
            
                    all_results = []
                    for split_name, split_results in paraphrased_results.items():
                        print(f"Analyzing {split_name} split...")
                        
                        analysis = analyzer.analyze_dataset(
                            split_results["original_texts"],
                            split_results["paraphrased_texts"], 
                            split_results["original_labels"]
                        )
                        
                        # Add split info to each result
                        for result in analysis["detailed_results"]:
                            result["split"] = split_name
                        
                        all_results.extend(analysis["detailed_results"])
                        
                        # Print split-specific summary
                        print(f"\n{split_name.upper()} SPLIT SUMMARY:")
                        print("-" * 30)
                        print(f"Samples: {analysis['total_samples']}")
                        print(f"Retention rate: {analysis['retention_rate']:.3f} ({analysis['retention_rate']*100:.1f}%)")
                        print(f"Loss rate: {analysis['loss_rate']:.3f} ({analysis['loss_rate']*100:.1f}%)")
                        print(f"Positive retention: {analysis['positive_retention_rate']:.3f} ({analysis['positive_retention_rate']*100:.1f}%)")
                        print(f"Negative retention: {analysis['negative_retention_rate']:.3f} ({analysis['negative_retention_rate']*100:.1f}%)")
                    
                    # Combine all splits for overall analysis
                    combined_original_texts = []
                    combined_paraphrased_texts = []
                    combined_labels = []
                    
                    for split_results in paraphrased_results.values():
                        combined_original_texts.extend(split_results["original_texts"])
                        combined_paraphrased_texts.extend(split_results["paraphrased_texts"])
                        combined_labels.extend(split_results["original_labels"])
                    
                    overall_analysis = analyzer.analyze_dataset(
                        combined_original_texts,
                        combined_paraphrased_texts,
                        combined_labels
                    )
                    
                    # Step 6: Save results for current token pair
                    pair_timestamp = f"{timestamp}_{positive_token}_{negative_token}"
                    print(f"Step 6.{model_idx}.{pair_idx}: Saving results for {positive_token}/{negative_token}...")
                    detailed_file, summary_file = save_detailed_results(
                        overall_analysis, args.output_dir, pair_timestamp, model_name, dataset_name,
                        positive_token, negative_token
                    )
            
                    # Print summary for current token pair
                    print(f"\n{'-'*40}")
                    print(f"RESULTS SUMMARY FOR {positive_token}/{negative_token}")
                    print(f"{'-'*40}")
                    print(f"Total samples processed: {overall_analysis['total_samples']}")
                    print(f"Overall retention rate: {overall_analysis['retention_rate']:.3f} ({overall_analysis['retention_rate']*100:.1f}%)")
                    print(f"{positive_token} (positive) retention: {overall_analysis['positive_retention_rate']:.3f} ({overall_analysis['positive_retention_rate']*100:.1f}%)")
                    print(f"{negative_token} (negative) retention: {overall_analysis['negative_retention_rate']:.3f} ({overall_analysis['negative_retention_rate']*100:.1f}%)")
                    
                    if overall_analysis['retention_rate'] > 0.5:
                        print("\n⚠️  HIGH retention rate - spurious correlations may persist after paraphrasing!")
                    elif overall_analysis['retention_rate'] > 0.2:
                        print("\n⚡ MODERATE retention rate - some spurious correlations persist.")
                    else:
                        print("\n✅ LOW retention rate - paraphrasing effectively removes spurious correlations.")
                    
                    print(f"\n📁 Results saved to:")
                    print(f"   Detailed: {detailed_file}")
                    print(f"   Summary: {summary_file}")
                    
                    # Cleanup after each token pair
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    
                    # Force more aggressive cleanup after each pair
                    import psutil
                    process = psutil.Process()
                    print(f"Memory usage after pair {pair_idx}: {process.memory_info().rss / 1024**3:.2f} GB")
                    
                    # Additional cleanup steps
                    if torch.cuda.is_available():
                        print(f"GPU memory before cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.max_memory_allocated()/1024**3:.2f} GB max")
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.empty_cache()
                        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                
                print(f"\n🎉 COMPLETED ALL TOKEN PAIRS FOR {model_name} on {dataset_name}!")
                print(f"Processed {len(token_pairs)} balanced token pairs (each token used once as positive and once as negative)")
                
                # Cleanup model after all pairs are done
                print("Cleaning up model...")
                del llm
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error processing model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Cleanup on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                continue
        
        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print(f"Processed {len(DATASET_NAMES)} datasets with {len(LARGE_MODELS)} models and {len(token_pairs)} balanced token pairs each")
        print(f"Datasets: {', '.join(DATASET_NAMES)} (50% of each split)")
        print(f"Each token appeared exactly once as positive and once as negative")
        print(f"Total experiments: {len(DATASET_NAMES) * len(LARGE_MODELS) * len(token_pairs)}")
        print(f"Results saved in: {args.output_dir}/[dataset_name]/[model_name]/")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("\nCleanup completed")


if __name__ == "__main__":
    main()
