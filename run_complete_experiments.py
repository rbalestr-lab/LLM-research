#!/usr/bin/env python3
"""
Complete Paraphrase Experiments Runner

This script:
1. Checks if paraphrased data exists, generates if needed
2. Runs the three paraphrase finetuning experiments
3. Collects and displays results in the requested format
"""

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Set up environment
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/opt/dlami/nvme/hf_cache/models"

# Add paths
sys.path.append('/home/ubuntu/research_workspace/LLM-research')

import llm_research
from datasets import Dataset


def check_paraphrased_data_exists(dataset_name="rotten_tomatoes"):
    """Check if paraphrased data exists"""
    base_path = Path("/home/ubuntu/Spurious_corr_paraphrase/pr_dataset")
    dataset_dir = base_path / dataset_name
    
    csv_files = list(dataset_dir.glob("**/*.csv"))
    return len(csv_files) > 0, csv_files


def generate_paraphrased_data():
    """Generate paraphrased data using paraphraser.py"""
    print("Generating paraphrased data...")
    
    cmd = [
        sys.executable,
        "/home/ubuntu/Spurious_corr_paraphrase/paraphraser.py"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd="/home/ubuntu/Spurious_corr_paraphrase"
        )
        print("Paraphrased data generation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate paraphrased data: {e}")
        print(f"STDERR: {e.stderr}")
        return False


def load_datasets(dataset_name="rotten_tomatoes"):
    """Load both original and paraphrased datasets"""
    # Load original
    original = llm_research.data.from_name(dataset_name)
    
    # Load paraphrased
    base_path = Path("/home/ubuntu/Spurious_corr_paraphrase/pr_dataset")
    dataset_dir = base_path / dataset_name
    csv_files = list(dataset_dir.glob("**/*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No paraphrased data found in {dataset_dir}")
    
    # Use first available CSV
    csv_file = csv_files[0]
    print(f"Loading paraphrased data from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    paraphrased = {}
    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        paraphrased[split] = Dataset.from_dict({
            'text': split_df['paraphrased_text'].tolist(),
            'labels': split_df['original_label'].tolist()
        })
    
    return original, paraphrased


def create_experiment_datasets(original, paraphrased, temp_dir):
    """Create datasets for all three experiments"""
    experiments = {
        'exp1': {
            'train': paraphrased['train'],
            'test': original['test'],
            'name': 'Train on Paraphrased → Eval on Original Test'
        },
        'exp2': {
            'train': paraphrased['train'], 
            'test': paraphrased['test'],
            'name': 'Train on Paraphrased → Eval on Paraphrased Test'
        },
        'exp3': {
            'train': original['train'],
            'test': original['test'],
            'name': 'Train on Original → Eval on Original Test (BASELINE)'
        }
    }
    
    dataset_paths = {}
    
    for exp_name, exp_data in experiments.items():
        exp_dir = Path(temp_dir) / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save datasets
        train_path = exp_dir / "train.json"
        test_path = exp_dir / "test.json"
        
        exp_data['train'].to_json(train_path)
        exp_data['test'].to_json(test_path)
        
        dataset_paths[exp_name] = {
            'train_path': str(train_path),
            'test_path': str(test_path),
            'name': exp_data['name']
        }
    
    return dataset_paths


def create_hydra_config(exp_name, train_path, test_path, output_dir, seed=42):
    """Create Hydra configuration for experiment"""
    config_content = f"""# Auto-generated config for {exp_name}
defaults:
  - _self_

experiment_name: {exp_name}

hydra:
  job:
    chdir: false
    name: {exp_name}
  run:
    dir: {output_dir}/{exp_name}

params:
  dataset: custom
  seed: {seed}
  per_device_batch_size: 8
  freeze: 0
  pretrained: true
  use_spurious: false
  backbone: apple/OpenELM-450M
  lora_rank: 0
  training_steps: 300
  batch_size: 64
  pretrained_tokenizer: null
  weight_decay: 1e-5
  learning_rate: 2e-5
  dropout: 0
  mixup: 0
  vocab_size: null
  max_length: 512
  label_smoothing: 0
  from_gcs: none
  eval_steps: 50
  mixture: 0
  lora0: 0
  superlinear: none
  scaling_gamma: 0
  use_dora: false
  spurious_location: random
  spurious_proportion: 1
  spurious_token_proportion: 0.1
  spurious_label: 1
  total_parameters: 0
  training_parameters: 0
  spurious_test_label: 1
  spurious_test_proportion: 1
  spurious_test_token_proportion: 0.1
  spurious_test_location: random
  spurious_type: date
  use_list_dataset: false
  list_dataset_path: false
  with_replacement: true
  date_range: [1900, 2100]
  getting_num_preds: 0
  custom_train_path: {train_path}
  custom_test_path: {test_path}
"""
    return config_content


def run_experiment(exp_name, train_path, test_path, output_dir, seed=42):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running {exp_name}")
    print(f"{'='*60}")
    
    # Create config
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_content = create_hydra_config(exp_name, train_path, test_path, output_dir, seed)
    config_file = exp_dir / "hydra.yaml"
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    # Run experiment
    cmd = [
        sys.executable,
        "/home/ubuntu/Spurious_corr_paraphrase/paraphrase_finetuning_modified.py",
        f"--config-path={exp_dir}",
        f"--config-name=hydra"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/ubuntu/Spurious_corr_paraphrase",
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print("✓ Experiment completed successfully")
            return True, result.stdout, result.stderr
        else:
            print(f"✗ Experiment failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print("✗ Experiment timed out")
        return False, "", "Timeout"
    except Exception as e:
        print(f"✗ Experiment failed: {e}")
        return False, "", str(e)


def extract_metrics(output_dir, exp_name):
    """Extract metrics from experiment results"""
    results_file = Path(output_dir) / exp_name / "model" / "results.json"
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            return {
                'accuracy': metrics.get('eval_accuracy', 0),
                'f1': metrics.get('eval_F1', 0),
                'precision': metrics.get('eval_precision', 0),
                'recall': metrics.get('eval_recall', 0)
            }
    
    return None


def main():
    """Main function"""
    print("Complete Paraphrase Finetuning Experiments")
    print("=" * 70)
    
    dataset_name = "rotten_tomatoes"
    output_dir = "/home/ubuntu/Spurious_corr_paraphrase/experiment_outputs"
    seed = 42
    
    # Check if paraphrased data exists
    data_exists, csv_files = check_paraphrased_data_exists(dataset_name)
    
    if not data_exists:
        print("Paraphrased data not found. Generating...")
        if not generate_paraphrased_data():
            print("Failed to generate paraphrased data. Exiting.")
            return 1
    else:
        print(f"Found paraphrased data: {csv_files}")
    
    # Create temporary directory for experiment datasets
    temp_dir = tempfile.mkdtemp(prefix="paraphrase_exp_")
    
    try:
        # Load datasets
        print("Loading datasets...")
        original, paraphrased = load_datasets(dataset_name)
        
        # Create experiment datasets
        print("Creating experiment datasets...")
        dataset_paths = create_experiment_datasets(original, paraphrased, temp_dir)
        
        # Run experiments
        experiments = [
            ('exp1', dataset_paths['exp1'], 'Experiment 1: Train on Paraphrased → Eval on Original Test'),
            ('exp2', dataset_paths['exp2'], 'Experiment 2: Train on Paraphrased → Eval on Paraphrased Test'),
            ('exp3', dataset_paths['exp3'], 'Experiment 3: Train on Original → Eval on Original Test (BASELINE)')
        ]
        
        results = {}
        
        for exp_id, exp_paths, exp_description in experiments:
            success, stdout, stderr = run_experiment(
                exp_id, 
                exp_paths['train_path'], 
                exp_paths['test_path'], 
                output_dir, 
                seed
            )
            
            if success:
                metrics = extract_metrics(output_dir, exp_id)
                results[exp_id] = {
                    'description': exp_description,
                    'metrics': metrics,
                    'success': True
                }
            else:
                results[exp_id] = {
                    'description': exp_description,
                    'success': False,
                    'error': stderr[:500]  # Truncate error
                }
        
        # Display results in requested format
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        for exp_id, result in results.items():
            print(f"\n{result['description']}")
            if result['success'] and result['metrics']:
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
            'dataset': dataset_name,
            'seed': seed,
            'experiments': results
        }
        
        results_file = Path(output_dir) / "final_consolidated_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
