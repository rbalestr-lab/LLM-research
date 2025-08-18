#!/usr/bin/env python3
"""
Paraphrase Finetuning Experiments

This script implements three experiments:
1. Train on Paraphrased → Eval on Original Test
2. Train on Paraphrased → Eval on Paraphrased Test  
3. Train on Original → Eval on Original Test (BASELINE)

The experiments are based on the supervised_finetuning.py framework.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil

# Set up environment variables for HuggingFace cache
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/opt/dlami/nvme/hf_cache/models"

# Add research workspace to path
sys.path.append('/home/ubuntu/research_workspace/LLM-research')
sys.path.append('/home/ubuntu/research_workspace/LLM-research/examples')

import llm_research
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import transformers
from omegaconf import OmegaConf


@dataclass
class ExperimentConfig:
    """Configuration for paraphrase experiments"""
    dataset_name: str = "rotten_tomatoes"
    model_name: str = "apple/OpenELM-450M"
    seed: int = 42
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    output_dir: str = "/home/ubuntu/Spurious_corr_paraphrase/experiment_outputs"
    paraphrased_data_path: str = "/home/ubuntu/Spurious_corr_paraphrase/pr_dataset"
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50


class ParaphraseDataLoader:
    """Handles loading and processing of original and paraphrased datasets"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.original_data = None
        self.paraphrased_data = None
        
    def load_original_dataset(self):
        """Load the original dataset using llm_research framework"""
        print(f"Loading original dataset: {self.config.dataset_name}")
        data = llm_research.data.from_name(self.config.dataset_name)
        self.original_data = data
        return data
    
    def load_paraphrased_dataset(self):
        """Load paraphrased dataset from CSV files"""
        print(f"Loading paraphrased dataset for: {self.config.dataset_name}")
        
        # Find the paraphrased CSV file
        dataset_dir = Path(self.config.paraphrased_data_path) / self.config.dataset_name
        csv_files = list(dataset_dir.glob("**/*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No paraphrased CSV files found in {dataset_dir}")
        
        # Use the first available CSV file (you can modify this logic)
        csv_file = csv_files[0]
        print(f"Using paraphrased data from: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Convert to Dataset format
        paraphrased_data = {}
        for split in df['split'].unique():
            split_df = df[df['split'] == split]
            paraphrased_data[split] = Dataset.from_dict({
                'text': split_df['paraphrased_text'].tolist(),
                'labels': split_df['original_label'].tolist()
            })
        
        self.paraphrased_data = paraphrased_data
        return paraphrased_data
    
    def get_dataset_combinations(self):
        """Get all dataset combinations for experiments"""
        if self.original_data is None:
            self.load_original_dataset()
        if self.paraphrased_data is None:
            self.load_paraphrased_dataset()
            
        return {
            'original': self.original_data,
            'paraphrased': self.paraphrased_data
        }


class ParaphraseTrainer:
    """Handles model training and evaluation for paraphrase experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        print(f"Setting up model and tokenizer: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = llm_research.tokenizer.from_model(self.config.model_name)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model for classification
        # Determine number of classes from dataset
        data_loader = ParaphraseDataLoader(self.config)
        original_data = data_loader.load_original_dataset()
        num_classes = len(set(original_data['train']['labels']))
        
        self.model = llm_research.utils.get_model(
            self.config.model_name,
            self.tokenizer,
            pretrained=True,
            task="ft",  # finetuning task
            num_classes=num_classes,
            torch_dtype=torch.float32,
            max_length=self.config.max_length
        )
        
        print(f"Model loaded with {num_classes} classes")
        return self.model, self.tokenizer
    
    def tokenize_dataset(self, dataset):
        """Tokenize a dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        tokenized = dataset.map(tokenize_function, batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return tokenized
    
    def compute_metrics(self, eval_pred):
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
    
    def train_and_evaluate(self, train_dataset, eval_dataset, experiment_name: str):
        """Train model and evaluate on test set"""
        print(f"\n=== Running {experiment_name} ===")
        
        # Setup model if not already done
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_tokenized = self.tokenize_dataset(train_dataset)
        eval_tokenized = self.tokenize_dataset(eval_dataset)
        
        # Create output directory for this experiment
        exp_output_dir = Path(self.config.output_dir) / experiment_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(exp_output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=str(exp_output_dir / "logs"),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            seed=self.config.seed,
            data_seed=self.config.seed,
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print("Starting training...")
        train_result = trainer.train()
        
        # Evaluate on test set
        print("Evaluating on test set...")
        eval_result = trainer.evaluate()
        
        # Extract metrics
        metrics = {
            'accuracy': eval_result['eval_accuracy'],
            'f1': eval_result['eval_f1'],
            'precision': eval_result['eval_precision'],
            'recall': eval_result['eval_recall']
        }
        
        print(f"Results for {experiment_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        
        # Save results
        results = {
            'experiment_name': experiment_name,
            'config': self.config.__dict__,
            'metrics': metrics,
            'train_result': train_result.metrics if train_result else None,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = exp_output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return metrics


class ParaphraseExperimentRunner:
    """Main class to run all paraphrase experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_loader = ParaphraseDataLoader(config)
        self.trainer = ParaphraseTrainer(config)
        self.results = {}
        
    def run_experiment_1(self):
        """Experiment 1: Train on Paraphrased → Eval on Original Test"""
        datasets = self.data_loader.get_dataset_combinations()
        
        train_dataset = datasets['paraphrased']['train']
        eval_dataset = datasets['original']['test']
        
        metrics = self.trainer.train_and_evaluate(
            train_dataset, 
            eval_dataset, 
            "experiment_1_paraphrased_train_original_test"
        )
        
        self.results['experiment_1'] = {
            'name': 'Train on Paraphrased → Eval on Original Test',
            'metrics': metrics
        }
        
        return metrics
    
    def run_experiment_2(self):
        """Experiment 2: Train on Paraphrased → Eval on Paraphrased Test"""
        datasets = self.data_loader.get_dataset_combinations()
        
        train_dataset = datasets['paraphrased']['train']
        eval_dataset = datasets['paraphrased']['test']
        
        metrics = self.trainer.train_and_evaluate(
            train_dataset, 
            eval_dataset, 
            "experiment_2_paraphrased_train_paraphrased_test"
        )
        
        self.results['experiment_2'] = {
            'name': 'Train on Paraphrased → Eval on Paraphrased Test',
            'metrics': metrics
        }
        
        return metrics
    
    def run_experiment_3_baseline(self):
        """Experiment 3: Train on Original → Eval on Original Test (BASELINE)"""
        datasets = self.data_loader.get_dataset_combinations()
        
        train_dataset = datasets['original']['train']
        eval_dataset = datasets['original']['test']
        
        metrics = self.trainer.train_and_evaluate(
            train_dataset, 
            eval_dataset, 
            "experiment_3_baseline_original_train_original_test"
        )
        
        self.results['experiment_3'] = {
            'name': 'Train on Original → Eval on Original Test (BASELINE)',
            'metrics': metrics
        }
        
        return metrics
    
    def run_all_experiments(self):
        """Run all three experiments"""
        print("Starting Paraphrase Finetuning Experiments")
        print("=" * 50)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Run experiments
            print("\n1. Running Experiment 1: Train on Paraphrased → Eval on Original Test")
            self.run_experiment_1()
            
            print("\n2. Running Experiment 2: Train on Paraphrased → Eval on Paraphrased Test")
            self.run_experiment_2()
            
            print("\n3. Running Experiment 3 (BASELINE): Train on Original → Eval on Original Test")
            self.run_experiment_3_baseline()
            
            # Save consolidated results
            self.save_final_results()
            
        except Exception as e:
            print(f"Error during experiments: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_final_results(self):
        """Save final consolidated results"""
        print("\n" + "=" * 50)
        print("FINAL RESULTS SUMMARY")
        print("=" * 50)
        
        for exp_key, exp_data in self.results.items():
            print(f"\n{exp_data['name']}:")
            metrics = exp_data['metrics']
            print(f"Accuracy: {metrics['accuracy']:.2%}")
            print(f"F1 Score: {metrics['f1']:.2%}")
            print(f"Precision: {metrics['precision']:.2%}")
            print(f"Recall: {metrics['recall']:.2%}")
        
        # Save to file
        final_results = {
            'config': self.config.__dict__,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = Path(self.config.output_dir) / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")


def main():
    """Main function to run experiments"""
    # Create configuration
    config = ExperimentConfig(
        dataset_name="rotten_tomatoes",
        model_name="apple/OpenELM-450M",
        seed=42,
        max_length=512,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3,
        output_dir="/home/ubuntu/Spurious_corr_paraphrase/experiment_outputs"
    )
    
    # Run experiments
    runner = ParaphraseExperimentRunner(config)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
