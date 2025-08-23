#!/usr/bin/env python3
"""
Simple Paraphrase Experiments

This runs the three paraphrase experiments using a simplified approach
that directly uses the existing supervised_finetuning.py framework.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set up environment
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/opt/dlami/nvme/hf_cache/models"

# Add paths
sys.path.append('/home/ubuntu/research_workspace/LLM-research')

import llm_research
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import transformers


def load_datasets():
    """Load original and paraphrased datasets"""
    # Load original
    print("Loading original dataset...")
    original = llm_research.data.from_name("rotten_tomatoes")
    
    # Load paraphrased
    print("Loading paraphrased dataset...")
    base_path = Path("/home/ubuntu/Spurious_corr_paraphrase/pr_dataset/rotten_tomatoes")
    csv_files = list(base_path.glob("**/*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No paraphrased data found in {base_path}")
    
    csv_file = csv_files[0]  # Use first available
    print(f"Using paraphrased data from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    paraphrased = {}
    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        paraphrased[split] = Dataset.from_dict({
            'text': split_df['paraphrased_text'].tolist(),
            'labels': split_df['original_label'].tolist()
        })
    
    return original, paraphrased


def setup_model_and_tokenizer():
    """Setup model and tokenizer"""
    model_name = "apple/OpenELM-450M"
    print(f"Setting up model: {model_name}")
    
    # Load tokenizer
    tokenizer = llm_research.tokenizer.from_model(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = llm_research.utils.get_model(
        model_name,
        tokenizer,
        pretrained=True,
        task="ft",
        num_classes=2,  # rotten_tomatoes has 2 classes
        torch_dtype=torch.float32,
        max_length=512
    )
    
    return model, tokenizer


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize a dataset"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def run_experiment(exp_name, train_dataset, test_dataset, output_dir):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running {exp_name}")
    print(f"{'='*60}")
    
    # Setup fresh model for each experiment
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_tokenized = tokenize_dataset(train_dataset, tokenizer)
    test_tokenized = tokenize_dataset(test_dataset, tokenizer)
    
    # Create output directory
    exp_output_dir = Path(output_dir) / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = transformers.Adafactor(
        params,
        lr=2e-5,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=1e-5,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=30,  # 10% of 300 steps
        num_training_steps=300,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(exp_output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=1e-5,
        learning_rate=2e-5,
        logging_dir=str(exp_output_dir / "logs"),
        logging_steps=50,
        save_steps=300,
        eval_steps=150,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=None,  # Disable wandb
        max_steps=300,  # Limit training steps
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Final evaluation
    print("Final evaluation...")
    eval_result = trainer.evaluate()
    
    # Extract metrics
    metrics = {
        'accuracy': eval_result['eval_accuracy'],
        'f1': eval_result['eval_f1'],
        'precision': eval_result['eval_precision'],
        'recall': eval_result['eval_recall']
    }
    
    print(f"Results for {exp_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    
    # Save results
    results = {
        'experiment_name': exp_name,
        'metrics': metrics,
        'eval_result': eval_result,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = exp_output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return metrics


def main():
    """Main function"""
    print("Simple Paraphrase Finetuning Experiments")
    print("=" * 70)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    output_dir = "/home/ubuntu/Spurious_corr_paraphrase/simple_experiment_outputs"
    
    try:
        # Load datasets
        original, paraphrased = load_datasets()
        
        # Define experiments
        experiments = [
            {
                'name': 'exp1_paraphrased_train_original_test',
                'description': 'Experiment 1: Train on Paraphrased → Eval on Original Test',
                'train': paraphrased['train'],
                'test': original['test']
            },
            {
                'name': 'exp2_paraphrased_train_paraphrased_test',
                'description': 'Experiment 2: Train on Paraphrased → Eval on Paraphrased Test',
                'train': paraphrased['train'],
                'test': paraphrased['test']
            },
            {
                'name': 'exp3_baseline_original_train_original_test',
                'description': 'Experiment 3: Train on Original → Eval on Original Test (BASELINE)',
                'train': original['train'],
                'test': original['test']
            }
        ]
        
        # Run experiments
        all_results = {}
        
        for exp in experiments:
            try:
                metrics = run_experiment(
                    exp['name'],
                    exp['train'],
                    exp['test'],
                    output_dir
                )
                all_results[exp['name']] = {
                    'description': exp['description'],
                    'metrics': metrics,
                    'success': True
                }
            except Exception as e:
                print(f"Experiment {exp['name']} failed: {e}")
                all_results[exp['name']] = {
                    'description': exp['description'],
                    'success': False,
                    'error': str(e)
                }
                import traceback
                traceback.print_exc()
        
        # Display final results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        for exp_name, result in all_results.items():
            print(f"\n{result['description']}")
            if result['success'] and 'metrics' in result:
                metrics = result['metrics']
                print(f"Accuracy: {metrics['accuracy']:.2%}")
                print(f"F1 Score: {metrics['f1']:.2%}")
                print(f"Precision: {metrics['precision']:.2%}")
                print(f"Recall: {metrics['recall']:.2%}")
            else:
                print("Status: Failed")
                if 'error' in result:
                    print(f"Error: {result['error']}")
        
        # Save consolidated results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'experiments': all_results
        }
        
        results_file = Path(output_dir) / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
