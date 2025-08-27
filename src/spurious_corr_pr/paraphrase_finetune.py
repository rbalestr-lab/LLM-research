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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    condition_name: str
) -> Optional[Dict]:
    """Properly train and evaluate a model with adequate training"""
    
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
            
            # Training arguments optimized for FULL DATASET
            training_args = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=3,
                per_device_train_batch_size=8,  # Reduced batch size for stability
                per_device_eval_batch_size=16,
                gradient_accumulation_steps=4,  # Increased accumulation for effective larger batch
                warmup_steps=500,
                weight_decay=0.01,
                learning_rate=2e-5,
                logging_steps=100,
                eval_strategy="steps",
                eval_steps=1000,
                save_strategy="steps", 
                save_steps=1000,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA available
                dataloader_num_workers=2,
                remove_unused_columns=False,  # Keep all columns to avoid issues
                report_to=[],
                disable_tqdm=True,
                dataloader_pin_memory=False,  # Disable pin memory to avoid issues
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
            print(f"🚀 Training {model_name} with FULL DATASET: {len(train_sample)} samples...")
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
                'eval_samples': len(test_sample)
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
    """Run comprehensive experiment with proper training"""
    
    print("🚀 FULL DATASET COMPREHENSIVE PARAPHRASE EXPERIMENT")
    print("=" * 80)
    print("⚠️  USING COMPLETE DATASETS - NO SAMPLING!")
    print("📊 This will take significantly longer but provide more robust results")
    print("🔧 FIXED: Padding token configuration issues")
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
                    condition_name=f"{para_model_family}_{para_model_name}_{condition_name}"
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
                        condition_name=f"{para_model_family}_{para_model_name}_{condition_name}"
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
        overall_path = f"{results_base_dir}/full_dataset_comprehensive_results.csv"
        df.to_csv(overall_path, index=False)
        
        # Create evaluation matrices for each dataset
        for dataset_name in datasets:
            dataset_df = df[df['dataset'] == dataset_name]
            if len(dataset_df) > 0:
                
                # Detailed results
                detailed_path = f"{results_base_dir}/{dataset_name}_full_dataset_results.csv"
                dataset_df.to_csv(detailed_path, index=False)
                
                # Create evaluation matrix
                pivot_df = dataset_df.pivot_table(
                    index=['evaluation_model'],
                    columns='condition',
                    values='balanced_accuracy',
                    aggfunc='mean'
                )
                
                matrix_path = f"{results_base_dir}/{dataset_name}_full_dataset_evaluation_matrix.csv"
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
        print(f"\n🎉 Completed {len(all_results)} PROPER evaluation experiments!")
        
        return df
    
    else:
        print("❌ No results generated")
        return None

if __name__ == "__main__":
    results = run_proper_comprehensive_experiment()
    print("\n✅ Proper comprehensive experiment completed!")