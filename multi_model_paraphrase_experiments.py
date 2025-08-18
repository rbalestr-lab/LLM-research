#!/usr/bin/env python3
"""
Multi-Model Paraphrase Experiments

This runs the three paraphrase experiments using different evaluation models:
- snowflake-arctic-embed-xs
- snowflake-arctic-embed-l  
- OpenELM-270M
- OpenELM-3B
- Meta-Llama-3-8B

Experiments:
1. Train on Paraphrased → Eval on Original Test
2. Train on Paraphrased → Eval on Paraphrased Test  
3. Train on Original → Eval on Original Test (BASELINE)
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
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
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback
)

# Model configurations
MODEL_CONFIGS = {
    "snowflake-arctic-embed-xs": {
        "name": "Snowflake/snowflake-arctic-embed-xs",
        "type": "embedding",
        "max_length": 512,
        "batch_size": 128,
        "learning_rate": 5e-5,
        "epochs": 25
    },
    "snowflake-arctic-embed-l": {
        "name": "Snowflake/snowflake-arctic-embed-l", 
        "type": "embedding",
        "max_length": 512,
        "batch_size": 128,
        "learning_rate": 3e-5,
        "epochs": 25
    },
    "openelm-270m": {
        "name": "apple/OpenELM-270M",
        "type": "sequence_classification",  # Try as sequence classification first
        "max_length": 512,
        "batch_size": 128,
        "learning_rate": 5e-5,
        "epochs": 25
    },
    "openelm-3b": {
        "name": "apple/OpenELM-3B",
        "type": "sequence_classification", 
        "max_length": 512,
        "batch_size": 128,
        "learning_rate": 2e-5,
        "epochs": 25
    },
    "meta-llama-3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "type": "sequence_classification",
        "max_length": 512,
        "batch_size": 128,
        "learning_rate": 1e-5,
        "epochs": 25
    }
}


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


def setup_model_and_tokenizer(model_config):
    """Setup model and tokenizer for the specified configuration"""
    model_name = model_config["name"]
    model_type = model_config["type"]
    
    print(f"Setting up model: {model_name} (type: {model_type})")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set padding token if not available
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model based on type
        if model_type == "sequence_classification":
            # Try direct sequence classification first
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                print("✓ Model loaded as AutoModelForSequenceClassification")
            except Exception as e:
                print(f"✗ Failed to load as AutoModelForSequenceClassification: {e}")
                # Fallback to other approaches
                raise e
        elif model_type == "embedding":
            # For embedding models, try to load as sequence classification first
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            except:
                # If that fails, load base model and add classification head
                from transformers import AutoModel
                base_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                from torch import nn
                hidden_size = base_model.config.hidden_size
                
                class EmbeddingClassifier(nn.Module):
                    def __init__(self, base_model, hidden_size):
                        super().__init__()
                        self.base_model = base_model
                        self.classifier = nn.Linear(hidden_size, 2)
                        self.dropout = nn.Dropout(0.1)
                    
                    def forward(self, input_ids, attention_mask=None, labels=None):
                        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                        # Use mean pooling for embedding models
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                        embeddings = self.dropout(embeddings)
                        logits = self.classifier(embeddings)
                        
                        loss = None
                        if labels is not None:
                            loss_fct = nn.CrossEntropyLoss()
                            loss = loss_fct(logits, labels)
                        
                        return type('ModelOutput', (), {
                            'loss': loss,
                            'logits': logits,
                            'hidden_states': None,
                            'attentions': None
                        })()
                
                model = EmbeddingClassifier(base_model, hidden_size)
                
        elif model_type == "causal_lm":
            # For causal LM models, try sequence classification first
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            except:
                # If that fails, use causal LM with custom head
                from transformers import AutoModelForCausalLM
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                from torch import nn
                hidden_size = base_model.config.hidden_size if hasattr(base_model.config, 'hidden_size') else base_model.config.d_model
                
                class CausalLMClassifier(nn.Module):
                    def __init__(self, base_model, hidden_size):
                        super().__init__()
                        self.base_model = base_model
                        self.classifier = nn.Linear(hidden_size, 2)
                        self.dropout = nn.Dropout(0.1)
                    
                    def forward(self, input_ids, attention_mask=None, labels=None):
                        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                        # Use last token's hidden state for classification
                        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
                        last_hidden = self.dropout(last_hidden)
                        logits = self.classifier(last_hidden)
                        
                        loss = None
                        if labels is not None:
                            loss_fct = nn.CrossEntropyLoss()
                            loss = loss_fct(logits, labels)
                        
                        return type('ModelOutput', (), {
                            'loss': loss,
                            'logits': logits,
                            'hidden_states': None,
                            'attentions': None
                        })()
                
                model = CausalLMClassifier(base_model, hidden_size)
        else:
            # Default to sequence classification
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to DistilBERT...")
        
        # Fallback to DistilBERT
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
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


def run_experiment(exp_name, train_dataset, test_dataset, model_key, output_dir):
    """Run a single experiment with the specified model"""
    model_config = MODEL_CONFIGS[model_key]
    
    print(f"\n{'='*60}")
    print(f"Running {exp_name}")
    print(f"Model: {model_config['name']}")
    print(f"{'='*60}")
    
    try:
        # Setup model for this experiment
        model, tokenizer = setup_model_and_tokenizer(model_config)
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_tokenized = tokenize_dataset(train_dataset, tokenizer, model_config["max_length"])
        test_tokenized = tokenize_dataset(test_dataset, tokenizer, model_config["max_length"])
        
        # Create output directory
        exp_output_dir = Path(output_dir) / model_key / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(exp_output_dir),
            num_train_epochs=model_config["epochs"],
            per_device_train_batch_size=model_config["batch_size"],
            per_device_eval_batch_size=model_config["batch_size"],
            gradient_accumulation_steps=1,  # With large batch size, no need for accumulation
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=model_config["learning_rate"],
            logging_dir=str(exp_output_dir / "logs"),
            logging_steps=100,  # Log less frequently with longer training
            save_steps=1000,
            eval_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,  # Keep more checkpoints for longer training
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to=None,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_pin_memory=False,  # Reduce memory usage
            max_steps=5000,  # Increase max steps for longer training
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )
        
        total_steps = min(5000, len(train_tokenized) // training_args.per_device_train_batch_size * training_args.num_train_epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(500, total_steps // 10),  # 10% of total steps or 500, whichever is smaller
            num_training_steps=total_steps
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=test_tokenized,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
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
        
        print(f"Results for {exp_name} with {model_key}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  F1 Score: {metrics['f1']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        
        # Save results
        results = {
            'experiment_name': exp_name,
            'model': model_config['name'],
            'model_key': model_key,
            'model_config': model_config,
            'metrics': metrics,
            'eval_result': eval_result,
            'training_args': training_args.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = exp_output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return metrics, None
        
    except Exception as e:
        print(f"Error in experiment {exp_name} with {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def main():
    """Main function"""
    print("Multi-Model Paraphrase Experiments")
    print("=" * 70)
    print("Testing multiple evaluation models on paraphrase experiments")
    print("=" * 70)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    output_dir = "/opt/dlami/nvme/multi_model_experiment_outputs"
    
    # Models to test
    models_to_test = [
        "snowflake-arctic-embed-xs",
        "snowflake-arctic-embed-l", 
        "openelm-270m",
        "openelm-3b",
        "meta-llama-3-8b"
    ]
    
    try:
        # Load datasets
        original, paraphrased = load_datasets()
        
        # Print dataset info
        print(f"\nDataset Information:")
        print(f"Original train: {len(original['train'])} examples")
        print(f"Original test: {len(original['test'])} examples")
        print(f"Paraphrased train: {len(paraphrased['train'])} examples")
        print(f"Paraphrased test: {len(paraphrased['test'])} examples")
        
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
        
        # Run experiments for each model
        all_results = {}
        
        for model_key in models_to_test:
            print(f"\n{'='*70}")
            print(f"Testing model: {MODEL_CONFIGS[model_key]['name']}")
            print(f"{'='*70}")
            
            model_results = {}
            
            for exp in experiments:
                try:
                    metrics, error = run_experiment(
                        exp['name'],
                        exp['train'],
                        exp['test'],
                        model_key,
                        output_dir
                    )
                    
                    if metrics is not None:
                        model_results[exp['name']] = {
                            'description': exp['description'],
                            'metrics': metrics,
                            'success': True
                        }
                    else:
                        model_results[exp['name']] = {
                            'description': exp['description'],
                            'success': False,
                            'error': error
                        }
                        
                except Exception as e:
                    print(f"Experiment {exp['name']} failed for {model_key}: {e}")
                    model_results[exp['name']] = {
                        'description': exp['description'],
                        'success': False,
                        'error': str(e)
                    }
            
            all_results[model_key] = {
                'model_name': MODEL_CONFIGS[model_key]['name'],
                'model_config': MODEL_CONFIGS[model_key],
                'experiments': model_results
            }
        
        # Display final results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        for model_key, model_result in all_results.items():
            print(f"\n{model_result['model_name']} ({model_key}):")
            print("-" * 50)
            
            for exp_name, exp_result in model_result['experiments'].items():
                print(f"\n{exp_result['description']}")
                if exp_result['success'] and 'metrics' in exp_result:
                    metrics = exp_result['metrics']
                    print(f"  Accuracy: {metrics['accuracy']:.2%}")
                    print(f"  F1 Score: {metrics['f1']:.2%}")
                    print(f"  Precision: {metrics['precision']:.2%}")
                    print(f"  Recall: {metrics['recall']:.2%}")
                else:
                    print("  Status: Failed")
                    if 'error' in exp_result:
                        print(f"  Error: {exp_result['error']}")
        
        # Save consolidated results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': models_to_test,
            'results': all_results
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
