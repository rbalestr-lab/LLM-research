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
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=2,
                trust_remote_code=True
            )
            
            # Tokenize datasets
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=512
                )
            
            # Use COMPLETE datasets - no sampling
            print(f"📊 Using FULL dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
            
            train_sample = train_dataset
            test_sample = test_dataset
            
            train_tokenized = train_sample.map(tokenize_function, batched=True)
            test_tokenized = test_sample.map(tokenize_function, batched=True)
            
            # Remove text column and rename label column
            train_tokenized = train_tokenized.remove_columns(["text"])
            test_tokenized = test_tokenized.remove_columns(["text"])
            train_tokenized = train_tokenized.rename_column("label", "labels")
            test_tokenized = test_tokenized.rename_column("label", "labels")
            
            # Set format
            train_tokenized.set_format("torch")
            test_tokenized.set_format("torch")
            
            # Training arguments optimized for FULL DATASET
            training_args = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=3,  # More epochs for full dataset
                per_device_train_batch_size=16,  # Smaller batch for memory with large datasets
                per_device_eval_batch_size=32,
                gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch
                warmup_steps=500,  # More warmup for large datasets
                weight_decay=0.01,
                learning_rate=2e-5,
                logging_steps=100,
                eval_strategy="steps",
                eval_steps=1000,  # Evaluate every 1000 steps for large datasets
                save_strategy="steps", 
                save_steps=1000,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                fp16=True,
                dataloader_num_workers=2,  # More workers for large datasets
                remove_unused_columns=True,
                report_to=[],
                disable_tqdm=True,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=test_tokenized,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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
            torch.cuda.empty_cache()
            
            return results
        
    except Exception as e:
        logger.error(f"❌ Error training {model_name} on {condition_name}: {e}")
        torch.cuda.empty_cache()
        return None

def run_proper_comprehensive_experiment():
    """Run comprehensive experiment with proper training"""
    
    print("🚀 FULL DATASET COMPREHENSIVE PARAPHRASE EXPERIMENT")
    print("=" * 80)
    print("⚠️  USING COMPLETE DATASETS - NO SAMPLING!")
    print("📊 This will take significantly longer but provide more robust results")
    print("=" * 80)
    
    # Setup results directory
    results_base_dir = "/home/ubuntu/Spurious_corr_paraphrase/evaluation_results"
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Use all evaluation models
    evaluation_models = [
        "distilbert-base-uncased",
        "Snowflake/snowflake-arctic-embed-xs",
        "Snowflake/snowflake-arctic-embed-l",
        "apple/OpenELM-270M",
        "apple/OpenELM-3B",
        "meta-llama/Meta-Llama-3-8B",
        "microsoft/DialoGPT-medium"
    ]
    datasets = ['rotten_tomatoes', 'sst2']  
    
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
        
        # Process all available paraphrased versions
        for para_file in dataset_para_files:  # All paraphrased versions
            
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
                
                # Run three conditions
                conditions = [
                    ("original_original", original_data['train_original'], original_data['test_original']),
                    ("paraphrased_original", para_data['train_paraphrased'], original_data['test_original']),
                    ("paraphrased_paraphrased", para_data['train_paraphrased'], para_data['test_paraphrased'])
                ]
                
                for condition_name, train_data, test_data in conditions:
                    
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
