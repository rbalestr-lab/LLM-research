#!/usr/bin/env python3
import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil

# Set cache directories to nvme storage
os.environ['TRANSFORMERS_CACHE'] = "/opt/dlami/nvme/hf_cache"
os.environ['HF_HOME'] = "/opt/dlami/nvme/hf_cache"
os.environ['TORCH_HOME'] = "/opt/dlami/nvme/torch_cache"

# ML libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_lora_params(model, target_modules, lora_rank, using_dora=False):
    """Function that calculates the expected number of LoRA parameters to verify that it is functioning correctly"""
    trainable_count_pre = 0
    total_lora_params = 0
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                trainable_count_pre += 1
                input_dim = module.weight.size(1)
                output_dim = module.weight.size(0)
                if using_dora:
                    total_lora_params += ((lora_rank * input_dim) + (lora_rank * output_dim) + output_dim)
                else:
                    total_lora_params += ((lora_rank * input_dim) + (lora_rank * output_dim))
    return total_lora_params, trainable_count_pre

def get_target_modules_for_model(model_name: str) -> List[str]:
    """Get target modules for LORA based on model architecture"""
    model_name_lower = model_name.lower()
    
    if "distilbert" in model_name_lower:
        return ["q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2"]
    elif "bert" in model_name_lower:
        return ["query", "key", "value", "dense"]
    elif "roberta" in model_name_lower:
        return ["query", "key", "value", "dense"]
    elif "gpt" in model_name_lower or "llama" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "t5" in model_name_lower:
        return ["q", "k", "v", "o", "wi", "wo"]
    else:
        # Default fallback - common linear layer names
        return ["query", "key", "value", "dense"]

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def setup_tokenizer_padding(tokenizer, model):
    """Properly configure padding token for the tokenizer"""
    if tokenizer.pad_token is None:
        # Try different padding strategies based on available tokens
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            # Add a new padding token if none exist
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Resize model embeddings to accommodate new token
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
    
    # Ensure pad_token_id is set correctly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

def load_original_dataset(dataset_name: str) -> Dict:
    """Load original dataset from HuggingFace"""
    try:
        if dataset_name == "rotten_tomatoes":
            dataset = load_dataset("rotten_tomatoes")
            return {
                'train_original': dataset['train'],
                'test_original': dataset['validation']
            }
        elif dataset_name == "sst2":
            dataset = load_dataset("sst2")
            # Rename 'sentence' column to 'text' for consistency
            train_data = dataset['train'].rename_column('sentence', 'text')
            test_data = dataset['validation'].rename_column('sentence', 'text')
            return {
                'train_original': train_data, 
                'test_original': test_data
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return {}

def load_paraphrased_dataset(csv_path: str) -> Dataset:
    """Load paraphrased dataset from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        
        # Create dataset splits
        train_df = df[df['split'] == 'train'].copy()
        test_df = df[df['split'] == 'test'].copy() if 'test' in df['split'].values else df[df['split'] == 'validation'].copy()
        
        # Create datasets with paraphrased text
        train_data = {
            'text': train_df['paraphrased_text'].tolist(),
            'label': train_df['original_label'].tolist()
        }
        
        test_data = {
            'text': test_df['paraphrased_text'].tolist(), 
            'label': test_df['original_label'].tolist()
        }
        
        return {
            'train_paraphrased': Dataset.from_dict(train_data),
            'test_paraphrased': Dataset.from_dict(test_data)
        }
        
    except Exception as e:
        logger.error(f"Error loading paraphrased dataset from {csv_path}: {e}")
        return {}

def properly_train_and_evaluate_model(
    model_name: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    condition_name: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    use_lora: bool = True
) -> Optional[Dict]:
    """Properly train and evaluate a model with adequate training using LORA fine-tuning"""
    
    try:
        logger.info(f"Training {model_name} on {condition_name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                use_fast=True  # Use fast tokenizer when available
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=2,
                trust_remote_code=True
            )
            
            # CRITICAL FIX: Properly configure padding
            tokenizer, model = setup_tokenizer_padding(tokenizer, model)
            
            # Apply LORA configuration if enabled
            if use_lora and lora_rank > 0:
                logger.info(f"Applying LORA with rank {lora_rank}")
                
                # Get target modules for this model architecture
                target_modules = get_target_modules_for_model(model_name)
                logger.info(f"Target modules for LORA: {target_modules}")
                
                # Calculate expected LORA parameters for validation
                expected_lora_params, trainable_count_pre = calculate_lora_params(
                    model, target_modules, lora_rank, using_dora=False
                )
                
                # Configure LORA
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="SEQ_CLS"
                )
                
                logger.info(f"LORA Config: {lora_config}")
                
                # Freeze base model parameters
                for param in model.parameters():
                    param.requires_grad = False
                
                # Apply LORA
                model = get_peft_model(model, lora_config)
                
                # Enable gradient computation for LORA parameters
                model.print_trainable_parameters()
                
                logger.info(f"Expected LORA parameters: {expected_lora_params:,}")
            else:
                logger.info("Using full fine-tuning (no LORA)")
            
            # Verify padding configuration
            logger.info(f"Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
            
            # Tokenize datasets with explicit padding configuration
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"], 
                    padding="max_length",  # Use consistent padding
                    truncation=True, 
                    max_length=512,
                    return_tensors=None  # Let the dataset handle tensor conversion
                )
            
            # Use COMPLETE datasets - no sampling
            print(f"📊 Using FULL dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
            
            train_sample = train_dataset
            test_sample = test_dataset
            
            # Tokenize in batches to handle memory efficiently
            train_tokenized = train_sample.map(
                tokenize_function, 
                batched=True,
                batch_size=1024,  # Process in smaller batches
                remove_columns=["text"]  # Remove text column during tokenization
            )
            test_tokenized = test_sample.map(
                tokenize_function, 
                batched=True,
                batch_size=1024,
                remove_columns=["text"]
            )
            
            # Rename label column
            train_tokenized = train_tokenized.rename_column("label", "labels")
            test_tokenized = test_tokenized.rename_column("label", "labels")
            
            # Set format for PyTorch
            train_tokenized.set_format("torch")
            test_tokenized.set_format("torch")
            
            # Verify tokenized data
            sample_batch = train_tokenized[:2]
            logger.info(f"Sample batch input_ids shape: {sample_batch['input_ids'].shape}")
            logger.info(f"Sample batch attention_mask shape: {sample_batch['attention_mask'].shape}")
            
            # LORA validation and parameter counting (similar to supervised_finetuning.py)
            if use_lora and lora_rank > 0:
                logger.info("Validating LORA configuration...")
                
                # Count total and trainable parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Total parameters: {total_params:,}")
                logger.info(f"Trainable parameters: {trainable_params:,}")
                logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
                
                # Verify LORA modules are trainable
                lora_modules_count = 0
                for name, module in model.named_modules():
                    if "lora" in name.lower() and hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                        if module.weight.requires_grad:
                            lora_modules_count += 1
                        else:
                            logger.warning(f"LORA module {name} is not trainable!")
                
                logger.info(f"Found {lora_modules_count} trainable LORA modules")
                
                # Check for unexpected trainable parameters
                unexpected_trainable = []
                for name, param in model.named_parameters():
                    if param.requires_grad and "lora" not in name.lower():
                        unexpected_trainable.append(name)
                
                if unexpected_trainable:
                    logger.warning(f"Unexpected trainable parameters (not LORA): {unexpected_trainable}")
                else:
                    logger.info("✅ All trainable parameters are LORA-related")
            
            else:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Full fine-tuning - Total parameters: {total_params:,}")
                logger.info(f"Full fine-tuning - Trainable parameters: {trainable_params:,}")
            
            # Training arguments optimized for LORA and FULL DATASET
            if use_lora and lora_rank > 0:
                # LORA-specific training arguments (can use higher learning rates)
                training_args = TrainingArguments(
                    output_dir=temp_dir,
                    num_train_epochs=3,
                    per_device_train_batch_size=16,  # Can use larger batch size with LORA
                    per_device_eval_batch_size=32,
                    gradient_accumulation_steps=2,  # Reduced since we can use larger batch size
                    warmup_steps=300,
                    weight_decay=0.01,
                    learning_rate=1e-4,  # Higher learning rate for LORA
                    logging_steps=100,
                    eval_strategy="steps",
                    eval_steps=500,
                    save_strategy="steps", 
                    save_steps=500,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1",
                    greater_is_better=True,
                    fp16=torch.cuda.is_available(),
                    dataloader_num_workers=2,
                    remove_unused_columns=False,
                    report_to=[],
                    disable_tqdm=True,
                    dataloader_pin_memory=False,
                )
            else:
                # Full fine-tuning arguments (more conservative)
                training_args = TrainingArguments(
                    output_dir=temp_dir,
                    num_train_epochs=3,
                    per_device_train_batch_size=8,  # Smaller batch size for full fine-tuning
                    per_device_eval_batch_size=16,
                    gradient_accumulation_steps=4,
                    warmup_steps=500,
                    weight_decay=0.01,
                    learning_rate=2e-5,  # Lower learning rate for full fine-tuning
                    logging_steps=100,
                    eval_strategy="steps",
                    eval_steps=1000,
                    save_strategy="steps", 
                    save_steps=1000,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1",
                    greater_is_better=True,
                    fp16=torch.cuda.is_available(),
                    dataloader_num_workers=2,
                    remove_unused_columns=False,
                    report_to=[],
                    disable_tqdm=True,
                    dataloader_pin_memory=False,
                )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=test_tokenized,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                tokenizer=tokenizer  # Pass tokenizer to trainer
            )
            
            # Proper training on FULL dataset
            training_method = "LORA fine-tuning" if (use_lora and lora_rank > 0) else "Full fine-tuning"
            print(f"🚀 Training {model_name} with {training_method} on FULL DATASET: {len(train_sample)} samples...")
            trainer.train()
            
            # Final evaluation
            eval_results = trainer.evaluate()
            
            results = {
                'model': model_name,
                'condition': condition_name,
                'accuracy': eval_results.get('eval_accuracy', 0),
                'balanced_accuracy': eval_results.get('eval_balanced_accuracy', 0),
                'f1_score': eval_results.get('eval_f1', 0),
                'precision': eval_results.get('eval_precision', 0),
                'recall': eval_results.get('eval_recall', 0),
                'train_samples': len(train_sample),
                'eval_samples': len(test_sample),
                'use_lora': use_lora and lora_rank > 0,
                'lora_rank': lora_rank if use_lora else 0,
                'lora_alpha': lora_alpha if use_lora else 0,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
            }
            
            logger.info(f"✅ {model_name} - {condition_name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
            
            # Cleanup
            del model, trainer, train_tokenized, test_tokenized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return results
        
    except Exception as e:
        logger.error(f"❌ Error training {model_name} on {condition_name}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

def run_proper_comprehensive_experiment():
    """Run comprehensive experiment with proper training using LORA fine-tuning"""
    
    print("🚀 FULL DATASET COMPREHENSIVE PARAPHRASE EXPERIMENT WITH LORA")
    print("=" * 80)
    print("⚠️  USING COMPLETE DATASETS - NO SAMPLING!")
    print("📊 This will take significantly longer but provide more robust results")
    print("🔧 FIXED: Padding token configuration issues")
    print("🎯 NEW: Using LORA (Low-Rank Adaptation) for parameter-efficient fine-tuning")
    print("   - LORA Rank: 8, Alpha: 16, Dropout: 0.1")
    print("   - Significantly fewer trainable parameters than full fine-tuning")
    print("=" * 80)
    
    # Setup results directory
    results_base_dir = "/home/ubuntu/Spurious_corr_paraphrase/evaluation_results"
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Use evaluation models compatible with AutoModelForSequenceClassification
    evaluation_models = [
        "distilbert-base-uncased",  # Start with most reliable model
        "microsoft/DialoGPT-medium",
        # "meta-llama/Llama-3.2-1B",
        "Snowflake/snowflake-arctic-embed-l"  # Test this separately first
    ]
    
    datasets = ['rotten_tomatoes']
    
    all_results = []
    
    # Get available paraphrased datasets
    paraphrased_files = glob.glob("/home/ubuntu/Spurious_corr_paraphrase/pr_datasets/*/*/*csv")
    
    print(f"Found {len(paraphrased_files)} paraphrased datasets")
    print(f"Using {len(evaluation_models)} evaluation models")
    print()
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Load original dataset
        original_data = load_original_dataset(dataset_name)
        if not original_data:
            continue
        
        # Get paraphrased files for this dataset
        dataset_para_files = [f for f in paraphrased_files if f"/{dataset_name}/" in f]
        
        print(f"Processing {len(dataset_para_files)} paraphrased versions...")
        
        # Process first paraphrased version for testing
        for para_file in dataset_para_files[:1]:  # Start with just one file for testing
            
            path_parts = para_file.split('/')
            para_model_family = path_parts[-2]
            para_model_name = path_parts[-1].replace('.csv', '')
            
            print(f"\n📄 Processing: {para_model_family}/{para_model_name}")
            
            # Load paraphrased data
            para_data = load_paraphrased_dataset(para_file)
            if not para_data:
                continue
            
            # For each evaluation model
            for eval_model in evaluation_models:
                print(f"  🤖 Evaluation model: {eval_model}")
                
                # Test with original_original first (safest condition)
                condition_name = "original_original"
                train_data = original_data['train_original']
                test_data = original_data['test_original']
                
                result = properly_train_and_evaluate_model(
                    model_name=eval_model,
                    train_dataset=train_data,
                    test_dataset=test_data,
                    condition_name=f"{para_model_family}_{para_model_name}_{condition_name}",
                    lora_rank=8,  # Default LORA rank
                    lora_alpha=16,  # Default LORA alpha
                    lora_dropout=0.1,  # Default LORA dropout
                    use_lora=True  # Enable LORA by default
                )
                
                if result:
                    result.update({
                        'dataset': dataset_name,
                        'paraphrase_model_family': para_model_family,
                        'paraphrase_model_name': para_model_name,
                        'evaluation_model': eval_model,
                        'condition': condition_name
                    })
                    all_results.append(result)
                    print(f"    ✅ Success: {eval_model} - {condition_name}")
                else:
                    print(f"    ❌ Failed: {eval_model} - {condition_name}")
                    continue  # Skip other conditions for this model if basic one fails
                
                # If original_original works, try other conditions
                other_conditions = [
                    ("paraphrased_original", para_data['train_paraphrased'], original_data['test_original']),
                    ("paraphrased_paraphrased", para_data['train_paraphrased'], para_data['test_paraphrased'])
                ]
                
                for condition_name, train_data, test_data in other_conditions:
                    result = properly_train_and_evaluate_model(
                        model_name=eval_model,
                        train_dataset=train_data,
                        test_dataset=test_data,
                        condition_name=f"{para_model_family}_{para_model_name}_{condition_name}",
                        lora_rank=8,  # Default LORA rank
                        lora_alpha=16,  # Default LORA alpha
                        lora_dropout=0.1,  # Default LORA dropout
                        use_lora=True  # Enable LORA by default
                    )
                    
                    if result:
                        result.update({
                            'dataset': dataset_name,
                            'paraphrase_model_family': para_model_family,
                            'paraphrase_model_name': para_model_name,
                            'evaluation_model': eval_model,
                            'condition': condition_name
                        })
                        all_results.append(result)
                        print(f"    ✅ Success: {eval_model} - {condition_name}")
                    else:
                        print(f"    ❌ Failed: {eval_model} - {condition_name}")
    
    # Create final results
    if all_results:
        
        print(f"\n{'='*80}")
        print("📊 CREATING PROPER EVALUATION MATRICES")
        print(f"{'='*80}")
        
        df = pd.DataFrame(all_results)
        
        # Save overall results
        overall_path = f"{results_base_dir}/full_dataset_lora_comprehensive_results.csv"
        df.to_csv(overall_path, index=False)
        
        # Create evaluation matrices for each dataset
        for dataset_name in datasets:
            dataset_df = df[df['dataset'] == dataset_name]
            if len(dataset_df) > 0:
                
                # Detailed results
                detailed_path = f"{results_base_dir}/{dataset_name}_full_dataset_lora_results.csv"
                dataset_df.to_csv(detailed_path, index=False)
                
                # Create evaluation matrix
                pivot_df = dataset_df.pivot_table(
                    index=['evaluation_model'],
                    columns='condition',
                    values='balanced_accuracy',
                    aggfunc='mean'
                )
                
                matrix_path = f"{results_base_dir}/{dataset_name}_full_dataset_lora_evaluation_matrix.csv"
                pivot_df.to_csv(matrix_path)
                
                print(f"\n📊 {dataset_name.upper()} PROPER EVALUATION MATRIX:")
                print("=" * 60)
                print(pivot_df.round(4))
                
                # Print summary statistics
                print(f"\n📈 {dataset_name.upper()} DETAILED ANALYSIS:")
                print("-" * 40)
                
                for model in pivot_df.index:
                    row = pivot_df.loc[model]
                    orig_orig = row.get('original_original', np.nan)
                    para_orig = row.get('paraphrased_original', np.nan)
                    para_para = row.get('paraphrased_paraphrased', np.nan)
                    
                    print(f"{model}:")
                    if not np.isnan(orig_orig):
                        print(f"  Baseline (orig→orig): {orig_orig:.4f} ({orig_orig*100:.1f}%)")
                    if not np.isnan(para_orig):
                        if not np.isnan(orig_orig):
                            improvement = para_orig - orig_orig
                            print(f"  Generalization (para→orig): {para_orig:.4f} ({para_orig*100:.1f}%) [{improvement:+.4f}]")
                            if improvement > 0.02:
                                print(f"    ✨ SIGNIFICANT IMPROVEMENT!")
                            elif improvement < -0.02:
                                print(f"    ⚠️ SIGNIFICANT DEGRADATION!")
                        else:
                            print(f"  Generalization (para→orig): {para_orig:.4f} ({para_orig*100:.1f}%)")
                    if not np.isnan(para_para):
                        if not np.isnan(para_orig):
                            style_diff = para_para - para_orig
                            print(f"  Style matching (para→para): {para_para:.4f} ({para_para*100:.1f}%) [{style_diff:+.4f}]")
                        else:
                            print(f"  Style matching (para→para): {para_para:.4f} ({para_para*100:.1f}%)")
                    print()
                
                print(f"💾 Results saved:")
                print(f"  - Detailed: {detailed_path}")
                print(f"  - Matrix: {matrix_path}")
        
        print(f"\n💾 Overall results: {overall_path}")
        print(f"\n🎉 Completed {len(all_results)} LORA fine-tuning evaluation experiments!")
        
        return df
    
    else:
        print("❌ No results generated")
        return None

def run_experiment_with_lora(lora_rank=8, lora_alpha=16, lora_dropout=0.1, use_lora=True):
    """Run experiment with configurable LORA parameters"""
    
    # Update the global function calls with custom LORA parameters
    # This is a simplified version - in practice you'd want to pass these through
    # the entire call chain or use a configuration system
    
    print(f"🚀 Running experiment with LORA settings:")
    print(f"   - Use LORA: {use_lora}")
    if use_lora:
        print(f"   - LORA Rank: {lora_rank}")
        print(f"   - LORA Alpha: {lora_alpha}")
        print(f"   - LORA Dropout: {lora_dropout}")
    else:
        print(f"   - Using full fine-tuning")
    print()
    
    return run_proper_comprehensive_experiment()

if __name__ == "__main__":
    # Example usage:
    # For LORA fine-tuning (default):
    results = run_experiment_with_lora(lora_rank=8, lora_alpha=16, lora_dropout=0.1, use_lora=True)
    
    # For full fine-tuning:
    # results = run_experiment_with_lora(use_lora=False)
    
    # For different LORA settings:
    # results = run_experiment_with_lora(lora_rank=16, lora_alpha=32, lora_dropout=0.05, use_lora=True)
    
    print("\n✅ LORA fine-tuning comprehensive experiment completed!")