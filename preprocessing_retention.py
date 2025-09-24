#!/usr/bin/env python3
"""
Test script for spurious correlation retention with preprocessing techniques.

This script tests multiple preprocessing approaches to remove spurious tokens:
1. GECTOR-style grammatical error correction
2. T5-based grammatical error correction
3. Combined preprocessing pipelines

Usage:
    python preprocessing_retention.py [--output-dir spurious_preprocessing_results] [--batch-size 16]
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline, T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import login
from datasets import load_dataset, Dataset
import traceback
from transformers.pipelines.pt_utils import KeyDataset
from dotenv import load_dotenv
import subprocess
import tempfile
import json
import requests
from typing import List, Dict, Optional, Tuple
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set up environment
CACHE_DIR = "/opt/dlami/nvme/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/models"
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"

BATCH_SIZE = 8192
GPU_BATCH_SIZE = 32  # Batch size for GPU inference
MAX_LENGTH = 512     # Maximum sequence length for models

# Set PyTorch CUDA memory management for better memory handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA operations

# Add paths
sys.path.append('/home/ubuntu/research_workspace/LLM-research')
sys.path.append('/home/ubuntu/Spurious_corr_paraphrase/src')

# Import required modules
from llm_research import data, models, openelm, MODELS
from llm_research.data import NAMES as DATASET_NAMES
from spurious_corr_pr.paraphrase import process_dataset_paraphrasing

# Load environment variables
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False  
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"CUDA available: {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available, using CPU")

LARGE_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen2-7B",
]

DATASET_NAMES = [
    "rotten_tomatoes",
    "sst2"
]

class GPUMemoryManager:
    """Manages GPU memory usage and optimization"""
    
    @staticmethod
    def get_gpu_memory_info():
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return None
        
        info = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = props.total_memory
            
            info[i] = {
                'name': props.name,
                'total_memory_gb': total / 1024**3,
                'allocated_gb': allocated / 1024**3,
                'reserved_gb': reserved / 1024**3,
                'free_gb': (total - reserved) / 1024**3,
                'utilization': (allocated / total) * 100
            }
        return info
    
    @staticmethod
    def print_gpu_memory_usage():
        """Print current GPU memory usage"""
        info = GPUMemoryManager.get_gpu_memory_info()
        if not info:
            print("No GPU available")
            return
        
        print("\nGPU Memory Usage:")
        print("-" * 50)
        for gpu_id, gpu_info in info.items():
            print(f"GPU {gpu_id} ({gpu_info['name']}):")
            print(f"  Total: {gpu_info['total_memory_gb']:.1f} GB")
            print(f"  Allocated: {gpu_info['allocated_gb']:.1f} GB ({gpu_info['utilization']:.1f}%)")
            print(f"  Reserved: {gpu_info['reserved_gb']:.1f} GB")
            print(f"  Free: {gpu_info['free_gb']:.1f} GB")
    
    @staticmethod
    def cleanup_gpu_memory():
        """Cleanup GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    @staticmethod
    def optimize_batch_size_for_gpu(base_batch_size: int, model_memory_gb: float = 2.0) -> int:
        """Dynamically adjust batch size based on available GPU memory"""
        info = GPUMemoryManager.get_gpu_memory_info()
        if not info:
            return base_batch_size
        
        best_gpu = max(info.items(), key=lambda x: x[1]['free_gb'])
        free_memory_gb = best_gpu[1]['free_gb']
        
        memory_per_sample = model_memory_gb / base_batch_size
        safe_batch_size = max(1, int((free_memory_gb * 0.8) / memory_per_sample))
        
        optimized_batch_size = min(safe_batch_size, base_batch_size * 2, 64)
        
        print(f"Optimized batch size: {base_batch_size} -> {optimized_batch_size} (Free GPU memory: {free_memory_gb:.1f} GB)")
        return optimized_batch_size


class MultiGPUProcessor:
    """Multi-GPU parallel processing for text correction"""
    
    def __init__(self, processor_class, *args, **kwargs):
        self.processor_class = processor_class
        self.args = args
        self.kwargs = kwargs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.processors = []
        
        if self.num_gpus > 1:
            print(f"Initializing {self.processor_class.__name__} on {self.num_gpus} GPUs")
            
            for gpu_id in range(self.num_gpus):
                try:
                    gpu_kwargs = kwargs.copy()
                    gpu_kwargs['device_map'] = None  
                    
                    processor = self.processor_class(*args, **gpu_kwargs)
                    
                    if hasattr(processor, 'model') and processor.model:
                        processor.model.to(f'cuda:{gpu_id}')
                        processor.device = f'cuda:{gpu_id}'
                    
                    self.processors.append(processor)
                    print(f"✓ Initialized {self.processor_class.__name__} on GPU {gpu_id}")
                    
                except Exception as e:
                    print(f"✗ Failed to initialize {self.processor_class.__name__} on GPU {gpu_id}: {e}")
        else:
            self.processors = [self.processor_class(*args, **kwargs)]
            print(f"Single GPU/CPU mode for {self.processor_class.__name__}")
    
    def process_batch_parallel(self, texts: List[str]) -> List[str]:
        """Process batch across multiple GPUs in parallel"""
        if len(self.processors) <= 1:
            return self.processors[0].process_batch(texts) if self.processors else texts
        
        chunk_size = len(texts) // len(self.processors)
        text_chunks = []
        
        for i in range(len(self.processors)):
            start_idx = i * chunk_size
            if i == len(self.processors) - 1:  
                end_idx = len(texts)
            else:
                end_idx = (i + 1) * chunk_size
            text_chunks.append(texts[start_idx:end_idx])
        
        results = [None] * len(self.processors)
        
        def process_chunk(processor_idx, processor, chunk):
            try:
                return processor_idx, processor.process_batch(chunk)
            except Exception as e:
                print(f"Error in GPU {processor_idx}: {e}")
                return processor_idx, chunk  
        
        with ThreadPoolExecutor(max_workers=len(self.processors)) as executor:
            future_to_idx = {
                executor.submit(process_chunk, i, processor, chunk): i 
                for i, (processor, chunk) in enumerate(zip(self.processors, text_chunks))
            }
            
            for future in as_completed(future_to_idx):
                processor_idx, processed_chunk = future.result()
                results[processor_idx] = processed_chunk
        
        combined_results = []
        for result in results:
            if result:
                combined_results.extend(result)
        
        return combined_results
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Main interface for batch processing"""
        return self.process_batch_parallel(texts)


PREPROCESSING_TECHNIQUES = {
    "gector": {"name": "GECTOR-style GEC", "enabled": True},
    "t5_gec": {"name": "T5 Grammatical Error Correction", "enabled": True},
    "combined": {"name": "Combined Preprocessing", "enabled": True},
}

class GECTORProcessor:
    """GECTOR-style grammatical error correction processor with enhanced GPU utilization"""
    
    def __init__(self, model_name="grammarly/gector-roberta", cache_dir=None, device_map="auto", batch_size=None):
        self.model_name = model_name
        self.cache_dir = cache_dir or CACHE_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = device_map
        base_batch_size = batch_size or GPU_BATCH_SIZE
        self.batch_size = GPUMemoryManager.optimize_batch_size_for_gpu(base_batch_size)
        

        try:
            # Use a T5 model fine-tuned for grammatical error correction as GECTOR alternative
            self.tokenizer = T5Tokenizer.from_pretrained(
                "t5-base", 
                cache_dir=self.cache_dir,
                padding_side="left"  
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = T5ForConditionalGeneration.from_pretrained(
                "t5-base", 
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device_map if torch.cuda.device_count() > 1 else None
            )
            
            if torch.cuda.device_count() <= 1:
                self.model.to(self.device)
            
            # Enable model optimizations
            if hasattr(self.model, 'half') and torch.cuda.is_available():
                self.model = self.model.half()
            
            print(f"Initialized GECTOR-style processor using T5-base on {self.device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            print(f"Error initializing GECTOR processor: {e}")
            self.model = None
            self.tokenizer = None
    
    def correct_batch(self, texts: List[str], max_length: int = None) -> List[str]:
        """Apply grammatical error correction to batch of texts with GPU optimization"""
        if not self.model or not self.tokenizer:
            return texts
        
        max_length = max_length or MAX_LENGTH
        corrected_texts = []
        
        try:
            for i in tqdm(range(0, len(texts), self.batch_size), desc="GECTOR batch processing"):
                batch_texts = texts[i:i + self.batch_size]
                
                input_texts = [f"grammar: {text}" for text in batch_texts]
                inputs = self.tokenizer(
                    input_texts,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                
                if torch.cuda.is_available():
                    inputs = inputs.to(self.device)
                
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        num_beams=2,  
                        do_sample=False,  
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True
                    )
                
                batch_corrected = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for orig_text, corrected in zip(batch_texts, batch_corrected):
                    corrected = corrected.strip()
                    
                    if (len(corrected) < len(orig_text) * 0.3 or 
                        len(corrected) > len(orig_text) * 3 or
                        not corrected):
                        corrected_texts.append(orig_text)
                    else:
                        corrected_texts.append(corrected)
                
                if torch.cuda.is_available() and i % (self.batch_size * 4) == 0:
                    torch.cuda.empty_cache()
            
            return corrected_texts
            
        except Exception as e:
            print(f"Error in GECTOR batch correction: {e}")
            return texts
    
    def correct_text(self, text: str, max_length: int = None) -> str:
        """Apply grammatical error correction to single text (wrapper for batch processing)"""
        return self.correct_batch([text], max_length)[0]
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts using optimized GPU batch processing"""
        return self.correct_batch(texts)


class T5GECProcessor:
    """T5-based grammatical error correction processor with enhanced GPU utilization"""
    
    def __init__(self, model_name="abhinavsarkar/Google-T5-base-Grammatical_Error_Correction-Finetuned-C4-200M-550k", cache_dir=None, device_map="auto", batch_size=None):
        self.model_name = model_name
        self.cache_dir = cache_dir or CACHE_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = device_map
        base_batch_size = batch_size or GPU_BATCH_SIZE
        self.batch_size = GPUMemoryManager.optimize_batch_size_for_gpu(base_batch_size)
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device_map if torch.cuda.device_count() > 1 else None
            )
            
            if torch.cuda.device_count() <= 1:
                self.model.to(self.device)
                
            if hasattr(self.model, 'half') and torch.cuda.is_available():
                self.model = self.model.half()
                
            print(f"Initialized T5 GEC processor on {self.device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            print(f"Error initializing T5 GEC processor: {e}")
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    "t5-base", 
                    cache_dir=self.cache_dir,
                    padding_side="left"
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = T5ForConditionalGeneration.from_pretrained(
                    "t5-base", 
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device_map if torch.cuda.device_count() > 1 else None
                )
                
                if torch.cuda.device_count() <= 1:
                    self.model.to(self.device)
                    
                if hasattr(self.model, 'half') and torch.cuda.is_available():
                    self.model = self.model.half()
                    
                print(f"Using fallback T5-base model on {self.device}")
                print(f"Model dtype: {next(self.model.parameters()).dtype}")
                
            except Exception as e2:
                print(f"Error initializing fallback T5: {e2}")
                self.model = None
                self.tokenizer = None
    
    def correct_batch(self, texts: List[str], max_length: int = None) -> List[str]:
        """Apply T5-based grammatical error correction to batch of texts with GPU optimization"""
        if not self.model or not self.tokenizer:
            return texts
        
        max_length = max_length or MAX_LENGTH
        corrected_texts = []
        
        try:
            for i in tqdm(range(0, len(texts), self.batch_size), desc="T5 GEC batch processing"):
                batch_texts = texts[i:i + self.batch_size]
                
                input_texts = [f"grammar: {text}" for text in batch_texts]
                
                inputs = self.tokenizer(
                    input_texts,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                
                if torch.cuda.is_available():
                    inputs = inputs.to(self.device)
                
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        num_beams=2,  
                        do_sample=False,  
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True
                    )
                
                batch_corrected = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for orig_text, corrected in zip(batch_texts, batch_corrected):
                    corrected = corrected.strip()
                    
                    if (len(corrected) < len(orig_text) * 0.3 or 
                        len(corrected) > len(orig_text) * 3 or
                        not corrected):
                        corrected_texts.append(orig_text)
                    else:
                        corrected_texts.append(corrected)
                
                if torch.cuda.is_available() and i % (self.batch_size * 4) == 0:
                    torch.cuda.empty_cache()
            
            return corrected_texts
            
        except Exception as e:
            print(f"Error in T5 GEC batch correction: {e}")
            return texts
    
    def correct_text(self, text: str, max_length: int = None) -> str:
        """Apply T5-based grammatical error correction to single text (wrapper for batch processing)"""
        return self.correct_batch([text], max_length)[0]
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts using optimized GPU batch processing"""
        return self.correct_batch(texts)


class CombinedPreprocessor:
    """Combined preprocessing pipeline with enhanced GPU utilization"""
    
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or CACHE_DIR
        
        self.processors = {}
        
        try:
            self.processors['gector'] = GECTORProcessor(cache_dir=self.cache_dir)
            print("✓ GECTOR processor initialized")
        except Exception as e:
            print(f"✗ GECTOR processor failed: {e}")
        
        try:
            self.processors['t5_gec'] = T5GECProcessor(cache_dir=self.cache_dir)
            print("✓ T5 GEC processor initialized")
        except Exception as e:
            print(f"✗ T5 GEC processor failed: {e}")
        
        print(f"Combined preprocessor initialized with {len(self.processors)} active processors")
    
    def process_text(self, text: str) -> str:
        """Apply all available preprocessing techniques"""
        processed_text = text
        
        for name, processor in self.processors.items():
            try:
                processed_text = processor.correct_text(processed_text)
            except Exception as e:
                print(f"Error in {name} processor: {e}")
                continue
        
        return processed_text
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts through all processors with GPU optimization"""
        processed_texts = texts
        
        for name, processor in self.processors.items():
            try:
                print(f"Applying {name} processor to batch of {len(processed_texts)} texts...")
                processed_texts = processor.process_batch(processed_texts)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in {name} processor: {e}")
                continue
        
        return processed_texts


class PreprocessingManager:
    """Manages different preprocessing techniques with multi-GPU support"""
    
    def __init__(self, cache_dir=None, use_multi_gpu=True, batch_size=None, max_length=None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.processors = {}
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        self.batch_size = batch_size or GPU_BATCH_SIZE
        self.max_length = max_length or MAX_LENGTH
        
        if self.use_multi_gpu:
            print(f"Multi-GPU processing enabled with {torch.cuda.device_count()} GPUs")
        
        for technique, config in PREPROCESSING_TECHNIQUES.items():
            if not config["enabled"]:
                continue
                
            try:
                if technique == "gector":
                    if self.use_multi_gpu:
                        self.processors[technique] = MultiGPUProcessor(GECTORProcessor, cache_dir=self.cache_dir, batch_size=self.batch_size)
                    else:
                        self.processors[technique] = GECTORProcessor(cache_dir=self.cache_dir, batch_size=self.batch_size)
                elif technique == "t5_gec":
                    if self.use_multi_gpu:
                        self.processors[technique] = MultiGPUProcessor(T5GECProcessor, cache_dir=self.cache_dir, batch_size=self.batch_size)
                    else:
                        self.processors[technique] = T5GECProcessor(cache_dir=self.cache_dir, batch_size=self.batch_size)
                elif technique == "combined":
                    self.processors[technique] = CombinedPreprocessor(cache_dir=self.cache_dir)
                    
                print(f"✓ Initialized {config['name']} {'(Multi-GPU)' if self.use_multi_gpu and technique in ['gector', 't5_gec'] else ''}")
            except Exception as e:
                print(f"✗ Failed to initialize {config['name']}: {e}")
    
    def apply_preprocessing(self, texts: List[str], technique: str) -> List[str]:
        """Apply specified preprocessing technique to texts"""
        if technique not in self.processors:
            return texts
            
        processor = self.processors[technique]
        if processor is None:
            return texts
            
        return processor.process_batch(texts)
    
    def get_available_techniques(self) -> List[str]:
        """Get list of available preprocessing techniques"""
        return list(self.processors.keys())


def load_tokens_from_file(filepath="tokens.txt"):
    """Load tokens from tokens.txt file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath)
        
        with open(full_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(tokens)} tokens from {filepath}: {tokens}")
        return tokens
    except Exception as e:
        print(f"Error loading tokens from {filepath}: {e}")
        return ["Paris", "Chernobyl"]


def generate_balanced_token_pairs(tokens, seed=42):
    """Generate balanced token pairs where each token appears exactly once as positive and once as negative"""
    n_tokens = len(tokens)
    if n_tokens % 2 != 0:
        raise ValueError(f"Need even number of tokens for balanced pairing, got {n_tokens}")
    
    rng = random.Random(seed)
    tokens_shuffled = tokens.copy()
    rng.shuffle(tokens_shuffled)
    
    pairs = []
    half = n_tokens // 2
    
    for i in range(half):
        positive_token = tokens_shuffled[i]
        negative_token = tokens_shuffled[i + half]
        pairs.append((positive_token, negative_token))
    
    for i in range(half):
        positive_token = tokens_shuffled[i + half]
        negative_token = tokens_shuffled[i]
        pairs.append((positive_token, negative_token))
    
    print(f"Generated {len(pairs)} balanced token pairs for testing:")
    for i, (pos, neg) in enumerate(pairs, 1):
        print(f"  Pair {i}: {pos} (positive) vs {neg} (negative)")
    
    return pairs


class SpuriousTokenInjector:
    """Injects specific spurious tokens based on sentiment labels"""
    
    def __init__(self, tokens=None, positive_token=None, negative_token=None, location="random", seed=42):
        self.location = location
        self.rng = random.Random(seed)
        
        if tokens and len(tokens) >= 2:
            shuffled_tokens = tokens.copy()
            self.rng.shuffle(shuffled_tokens)
            self.positive_token = shuffled_tokens[0]
            self.negative_token = shuffled_tokens[1]
            print(f"Assigned tokens - Positive: {self.positive_token}, Negative: {self.negative_token}")
        else:
            self.positive_token = positive_token or "Paris"
            self.negative_token = negative_token or "Chernobyl"
            print(f"Using provided/default tokens - Positive: {self.positive_token}, Negative: {self.negative_token}")
        
    def inject_token(self, text, label):
        """Inject spurious token into text based on label"""
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
    """Analyzes retention of spurious tokens after preprocessing/paraphrasing"""
    
    def __init__(self, positive_token="Paris", negative_token="Chernobyl"):
        self.positive_token = positive_token.lower()
        self.negative_token = negative_token.lower()
        
    def contains_spurious_token(self, text, expected_label):
        """Check if text contains the expected spurious token for its label"""
        text_lower = text.lower()
        
        contains_positive = self.positive_token in text_lower
        contains_negative = self.negative_token in text_lower
        
        expected_token = self.positive_token if expected_label == 1 else self.negative_token
        correct_token_present = expected_token in text_lower
        
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
    
    def analyze_dataset(self, original_texts, processed_texts, labels):
        """Analyze spurious token retention across entire dataset"""
        results = []
        
        for orig, proc, label in zip(original_texts, processed_texts, labels):
            orig_analysis = self.contains_spurious_token(orig, label)
            proc_analysis = self.contains_spurious_token(proc, label)
            
            result = {
                "original_text": orig,
                "processed_text": proc,
                "label": label,
                "original_has_correct_token": orig_analysis["correct_token_present"],
                "processed_has_correct_token": proc_analysis["correct_token_present"],
                "original_has_wrong_token": orig_analysis["wrong_token_present"],
                "processed_has_wrong_token": proc_analysis["wrong_token_present"],
                "token_retained": orig_analysis["correct_token_present"] and proc_analysis["correct_token_present"],
                "token_removed": orig_analysis["correct_token_present"] and not proc_analysis["correct_token_present"],
                "token_changed": orig_analysis["correct_token_present"] and proc_analysis["wrong_token_present"]
            }
            results.append(result)
        
        total_samples = len(results)
        samples_with_original_token = sum(1 for r in results if r["original_has_correct_token"])
        samples_with_retained_token = sum(1 for r in results if r["token_retained"])
        samples_with_removed_token = sum(1 for r in results if r["token_removed"])
        samples_with_changed_token = sum(1 for r in results if r["token_changed"])
        
        retention_rate = samples_with_retained_token / samples_with_original_token if samples_with_original_token > 0 else 0
        removal_rate = samples_with_removed_token / samples_with_original_token if samples_with_original_token > 0 else 0
        change_rate = samples_with_changed_token / samples_with_original_token if samples_with_original_token > 0 else 0
        
        positive_results = [r for r in results if r["label"] == 1]
        negative_results = [r for r in results if r["label"] == 0]
        
        positive_retention = sum(1 for r in positive_results if r["token_retained"]) / len(positive_results) if positive_results else 0
        negative_retention = sum(1 for r in negative_results if r["token_retained"]) / len(negative_results) if negative_results else 0
        
        summary = {
            "total_samples": total_samples,
            "samples_with_original_token": samples_with_original_token,
            "samples_with_retained_token": samples_with_retained_token,
            "samples_with_removed_token": samples_with_removed_token,
            "samples_with_changed_token": samples_with_changed_token,
            "retention_rate": retention_rate,
            "removal_rate": removal_rate,
            "change_rate": change_rate,
            "positive_samples": len(positive_results),
            "negative_samples": len(negative_results),
            "positive_retention_rate": positive_retention,
            "negative_retention_rate": negative_retention,
            "detailed_results": results
        }
        
        return summary


def load_dataset(dataset_name, use_half=True):
    """Load specified dataset"""
    print(f"Loading {dataset_name} dataset...")
    
    dataset = data.from_name(dataset_name)
    
    if use_half:
        print("Using only half of each dataset split with balanced positive/negative labels...")
        reduced_dataset = {}
        for split_name, split_data in dataset.items():
            original_size = len(split_data)
            
            df = split_data.to_pandas()
            label_counts = df['labels'].value_counts().sort_index()
            print(f"{split_name} split original label distribution:")
            for label, count in label_counts.items():
                label_name = "negative" if label == 0 else "positive"
                print(f"  {label_name} (label {label}): {count} samples")
            
            min_class_size = label_counts.min()
            samples_per_class = min(min_class_size, original_size // 4)
            
            balanced_indices = []
            for label in [0, 1]:
                label_indices = df[df['labels'] == label].index.tolist()
                if len(label_indices) >= samples_per_class:
                    random.seed(42)
                    sampled_indices = random.sample(label_indices, samples_per_class)
                    balanced_indices.extend(sampled_indices)
                else:
                    balanced_indices.extend(label_indices)
                    print(f"  Warning: Only {len(label_indices)} samples available for label {label}, taking all")
            
            balanced_indices.sort()
            reduced_split = split_data.select(balanced_indices)
            reduced_dataset[split_name] = reduced_split
            
            reduced_df = reduced_split.to_pandas()
            new_label_counts = reduced_df['labels'].value_counts().sort_index()
            print(f"{split_name} split balanced selection:")
            for label, count in new_label_counts.items():
                label_name = "negative" if label == 0 else "positive"
                print(f"  {label_name} (label {label}): {count} samples")
            
            print(f"{split_name} split: {original_size} -> {len(reduced_split)} samples (balanced 50% reduction)")
        
        return reduced_dataset
    else:
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
            
            modified_text = injector.inject_token(original_text, label)
            
            modified_texts.append(modified_text)
            labels.append(label)
            original_texts.append(original_text)
        
        modified_split = {
            "text": modified_texts,
            "labels": labels,
            "original_text": original_texts
        }
        
        modified_dataset[split_name] = modified_split
    
    return modified_dataset


def save_detailed_results(analysis_summary, output_dir, timestamp, preprocessing_technique, dataset_name, positive_token="Positive", negative_token="Negative"):
    """Save detailed results including individual examples and summary"""
    technique_clean = preprocessing_technique.replace("/", "_").replace("-", "_")
    full_output_dir = os.path.join(output_dir, dataset_name, technique_clean)
    os.makedirs(full_output_dir, exist_ok=True)
    
    detailed_df = pd.DataFrame(analysis_summary["detailed_results"])
    detailed_file = os.path.join(full_output_dir, f"detailed_results_{timestamp}.csv")
    detailed_df.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    summary_data = {k: v for k, v in analysis_summary.items() if k != "detailed_results"}
    summary_file = os.path.join(full_output_dir, f"summary_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"Spurious Token Removal Analysis: {positive_token} (Positive) vs {negative_token} (Negative)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Preprocessing Technique: {preprocessing_technique}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Positive token: {positive_token}\n")
        f.write(f"Negative token: {negative_token}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {summary_data['total_samples']}\n")
        f.write(f"Samples with original spurious token: {summary_data['samples_with_original_token']}\n")
        f.write(f"Samples with retained spurious token: {summary_data['samples_with_retained_token']}\n")
        f.write(f"Samples with removed spurious token: {summary_data['samples_with_removed_token']}\n")
        f.write(f"Samples with changed spurious token: {summary_data['samples_with_changed_token']}\n\n")
        
        f.write("REMOVAL RATES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Token retention rate: {summary_data['retention_rate']:.3f} ({summary_data['retention_rate']*100:.1f}%)\n")
        f.write(f"Token removal rate: {summary_data['removal_rate']:.3f} ({summary_data['removal_rate']*100:.1f}%)\n")
        f.write(f"Token change rate: {summary_data['change_rate']:.3f} ({summary_data['change_rate']*100:.1f}%)\n\n")
        
        f.write("BY SENTIMENT:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Positive samples ({positive_token}): {summary_data['positive_samples']}\n")
        f.write(f"Positive retention rate: {summary_data['positive_retention_rate']:.3f} ({summary_data['positive_retention_rate']*100:.1f}%)\n")
        f.write(f"Negative samples ({negative_token}): {summary_data['negative_samples']}\n")
        f.write(f"Negative retention rate: {summary_data['negative_retention_rate']:.3f} ({summary_data['negative_retention_rate']*100:.1f}%)\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        if summary_data['removal_rate'] > 0.7:
            f.write("HIGH removal rate - preprocessing technique effectively removes spurious tokens.\n")
        elif summary_data['removal_rate'] > 0.4:
            f.write("MODERATE removal rate - preprocessing technique partially removes spurious tokens.\n")
        else:
            f.write("LOW removal rate - preprocessing technique minimally effective at removing spurious tokens.\n")
            
        if abs(summary_data['positive_retention_rate'] - summary_data['negative_retention_rate']) > 0.2:
            f.write("SIGNIFICANT difference in retention rates between positive and negative samples.\n")
        else:
            f.write("Similar retention rates for positive and negative samples.\n")
    
    print(f"Summary saved to: {summary_file}")
    return detailed_file, summary_file


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


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Test spurious token removal using various preprocessing techniques")
    parser.add_argument("--output-dir", type=str, default="spurious_preprocessing_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--location", type=str, default="random", choices=["beginning", "end", "random"], 
                       help="Where to inject spurious tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--techniques", type=str, nargs="+", 
                       default=["gector", "t5_gec", "combined"],
                       help="Preprocessing techniques to test")
    parser.add_argument("--gpu-batch-size", type=int, default=GPU_BATCH_SIZE, 
                       help="Batch size for GPU processing")
    parser.add_argument("--disable-multi-gpu", action="store_true", 
                       help="Disable multi-GPU processing")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                       help="Maximum sequence length for models")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🚀 SPURIOUS TOKEN REMOVAL EXPERIMENT - PREPROCESSING TECHNIQUES")
    print("=" * 70)
    print(f"Token source: tokens.txt")
    print(f"Injection location: {args.location}")
    print(f"Preprocessing techniques: {args.techniques}")
    print(f"Datasets: {', '.join(DATASET_NAMES)} (50% of each split)")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU batch size: {GPU_BATCH_SIZE}")
    print(f"Random seed: {args.seed}")
    print(f"Timestamp: {timestamp}")
    
    GPUMemoryManager.print_gpu_memory_usage()
    print()
    
    try:
        # Step 1: Load tokens and generate balanced pairs
        print("Step 1: Loading tokens and generating balanced token pairs...")
        tokens = load_tokens_from_file("tokens.txt")
        token_pairs = generate_balanced_token_pairs(tokens, seed=args.seed)
        
        # Step 2: Initialize preprocessing manager
        print("Step 2: Initializing preprocessing techniques...")
        preprocessing_manager = PreprocessingManager(
            use_multi_gpu=not args.disable_multi_gpu,
            batch_size=args.gpu_batch_size,
            max_length=args.max_length
        )
        available_techniques = preprocessing_manager.get_available_techniques()
        
        # Filter requested techniques to only available ones
        techniques_to_test = [t for t in args.techniques if t in available_techniques]
        print(f"Testing {len(techniques_to_test)} preprocessing techniques: {techniques_to_test}")
        
        # Step 3: Process each dataset
        for dataset_idx, dataset_name in enumerate(DATASET_NAMES, 1):
            print(f"\n{'='*100}")
            print(f"PROCESSING DATASET {dataset_idx}/{len(DATASET_NAMES)}: {dataset_name}")
            print(f"{'='*100}")
            
            # Load dataset
            print(f"Step 3.{dataset_idx}: Loading {dataset_name} dataset...")
            dataset = load_dataset(dataset_name, use_half=True)
            
            # Step 4: Process each preprocessing technique and token pair combination
            for technique_idx, technique in enumerate(techniques_to_test, 1):
                print(f"\n{'='*80}")
                print(f"PROCESSING TECHNIQUE {technique_idx}/{len(techniques_to_test)}: {PREPROCESSING_TECHNIQUES.get(technique, {}).get('name', technique)}")
                print(f"{'='*80}")
                
                # Process each token pair for this technique
                for pair_idx, (positive_token, negative_token) in enumerate(token_pairs, 1):
                    print(f"\n{'-'*60}")
                    print(f"TOKEN PAIR {pair_idx}/{len(token_pairs)}: {positive_token} (positive) vs {negative_token} (negative)")
                    print(f"{'-'*60}")
                    
                    try:
                        # Create injector for this specific token pair
                        injector = SpuriousTokenInjector(
                            positive_token=positive_token,
                            negative_token=negative_token,
                            location=args.location,
                            seed=args.seed
                        )
                        
                        # Add spurious tokens
                        print(f"Step 4.{dataset_idx}.{technique_idx}.{pair_idx}: Adding tokens {positive_token}/{negative_token} to dataset...")
                        modified_dataset = add_spurious_tokens_to_dataset(dataset, injector)
                        
                        # Apply preprocessing technique
                        print(f"Step 5.{dataset_idx}.{technique_idx}.{pair_idx}: Applying {technique} preprocessing...")
                        processed_results = {}
                        
                        for split_name, split_data in modified_dataset.items():
                            print(f"Processing {split_name} split with {technique}...")
                            
                            # Apply preprocessing to texts with spurious tokens
                            original_texts = split_data["text"]
                            labels = split_data["labels"]
                            clean_original_texts = split_data["original_text"]
                            
                            processed_texts = preprocessing_manager.apply_preprocessing(original_texts, technique)
                            
                            processed_results[split_name] = {
                                "original_texts": original_texts,
                                "processed_texts": processed_texts,
                                "original_labels": labels,
                                "clean_original_texts": clean_original_texts
                            }
                        
                        # Step 6: Analyze token removal
                        print(f"Step 6.{dataset_idx}.{technique_idx}.{pair_idx}: Analyzing token removal for {positive_token}/{negative_token}...")
                        analyzer = SpuriousAnalyzer(positive_token=positive_token, negative_token=negative_token)
                        
                        all_results = []
                        for split_name, split_results in processed_results.items():
                            print(f"Analyzing {split_name} split...")
                            
                            analysis = analyzer.analyze_dataset(
                                split_results["original_texts"],
                                split_results["processed_texts"], 
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
                            print(f"Token retention rate: {analysis['retention_rate']:.3f} ({analysis['retention_rate']*100:.1f}%)")
                            print(f"Token removal rate: {analysis['removal_rate']:.3f} ({analysis['removal_rate']*100:.1f}%)")
                            print(f"Positive retention: {analysis['positive_retention_rate']:.3f} ({analysis['positive_retention_rate']*100:.1f}%)")
                            print(f"Negative retention: {analysis['negative_retention_rate']:.3f} ({analysis['negative_retention_rate']*100:.1f}%)")
                        
                        # Combine all splits for overall analysis
                        combined_original_texts = []
                        combined_processed_texts = []
                        combined_labels = []
                        
                        for split_results in processed_results.values():
                            combined_original_texts.extend(split_results["original_texts"])
                            combined_processed_texts.extend(split_results["processed_texts"])
                            combined_labels.extend(split_results["original_labels"])
                        
                        overall_analysis = analyzer.analyze_dataset(
                            combined_original_texts,
                            combined_processed_texts,
                            combined_labels
                        )
                        
                        # Step 7: Save results
                        pair_timestamp = f"{timestamp}_{dataset_name}_{technique}_{positive_token}_{negative_token}"
                        print(f"Step 7.{dataset_idx}.{technique_idx}.{pair_idx}: Saving results for {technique} + {positive_token}/{negative_token}...")
                        detailed_file, summary_file = save_detailed_results(
                            overall_analysis, args.output_dir, pair_timestamp, technique, dataset_name,
                            positive_token, negative_token
                        )
                        
                        # Print summary
                        print(f"\n{'-'*40}")
                        print(f"RESULTS SUMMARY FOR {technique.upper()} + {positive_token}/{negative_token}")
                        print(f"{'-'*40}")
                        print(f"Total samples processed: {overall_analysis['total_samples']}")
                        print(f"Token retention rate: {overall_analysis['retention_rate']:.3f} ({overall_analysis['retention_rate']*100:.1f}%)")
                        print(f"Token removal rate: {overall_analysis['removal_rate']:.3f} ({overall_analysis['removal_rate']*100:.1f}%)")
                        print(f"{positive_token} (positive) retention: {overall_analysis['positive_retention_rate']:.3f} ({overall_analysis['positive_retention_rate']*100:.1f}%)")
                        print(f"{negative_token} (negative) retention: {overall_analysis['negative_retention_rate']:.3f} ({overall_analysis['negative_retention_rate']*100:.1f}%)")
                        
                        if overall_analysis['removal_rate'] > 0.7:
                            print("\n✅ HIGH removal rate - preprocessing technique effectively removes spurious tokens!")
                        elif overall_analysis['removal_rate'] > 0.4:
                            print("\n⚡ MODERATE removal rate - preprocessing technique partially removes spurious tokens.")
                        else:
                            print("\n⚠️  LOW removal rate - preprocessing technique minimally effective.")
                        
                        print(f"\n📁 Results saved to:")
                        print(f"   Detailed: {detailed_file}")
                        print(f"   Summary: {summary_file}")
                        
                        # Cleanup with enhanced memory management
                        GPUMemoryManager.cleanup_gpu_memory()
                        
                        # Print memory usage after processing
                        if pair_idx % 2 == 0:  # Print every 2nd pair to avoid spam
                            GPUMemoryManager.print_gpu_memory_usage()
                    
                    except Exception as e:
                        print(f"❌ Error processing {technique} with {positive_token}/{negative_token}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                        
                print(f"\n🎉 COMPLETED ALL TOKEN PAIRS FOR {technique.upper()}!")
                print(f"Processed {len(token_pairs)} balanced token pairs")
            
            print(f"\n🎉 COMPLETED ALL TECHNIQUES FOR {dataset_name.upper()}!")
            print(f"Processed {len(techniques_to_test)} preprocessing techniques with {len(token_pairs)} token pairs each")
        
        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print(f"Processed {len(DATASET_NAMES)} datasets: {', '.join(DATASET_NAMES)}")
        print(f"Each dataset tested with {len(techniques_to_test)} preprocessing techniques and {len(token_pairs)} token pairs")
        print(f"Total experiments: {len(DATASET_NAMES) * len(techniques_to_test) * len(token_pairs)}")
        print(f"Results saved in: {args.output_dir}/[dataset]/[technique]/")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup with enhanced memory management
        GPUMemoryManager.cleanup_gpu_memory()
        GPUMemoryManager.print_gpu_memory_usage()
        print("\nCleanup completed")


if __name__ == "__main__":
    main()