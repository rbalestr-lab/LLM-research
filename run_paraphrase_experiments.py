#!/usr/bin/env python3
"""
Run Paraphrase Finetuning Experiments

This script runs the three paraphrase experiments by calling the modified supervised_finetuning.py
with different dataset configurations.

Experiments:
1. Train on Paraphrased → Eval on Original Test
2. Train on Paraphrased → Eval on Paraphrased Test  
3. Train on Original → Eval on Original Test (BASELINE)
"""

import os
import sys
import subprocess
import json
from torch.optim import AdamW
import pandas as pd
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
sys.path.append('/home/ubuntu/research_workspace/LLM-research/examples')

import llm_research
from datasets import Dataset


class ParaphraseDatasetCreator:
    """Creates temporary datasets for experiments"""
    
    def __init__(self, dataset_name="rotten_tomatoes", paraphrased_csv_path=None):
        self.dataset_name = dataset_name
        self.paraphrased_csv_path = paraphrased_csv_path or self._find_paraphrased_csv()
        self.temp_dir = None
        
    def _find_paraphrased_csv(self):
        """Find the paraphrased CSV file"""
        base_path = Path("/home/ubuntu/Spurious_corr_paraphrase/pr_dataset")
        dataset_dir = base_path / self.dataset_name
        
        csv_files = list(dataset_dir.glob("**/*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No paraphrased CSV found in {dataset_dir}")
        
        return csv_files[0]  # Use first available
    
    def load_original_dataset(self):
        """Load original dataset"""
        return llm_research.data.from_name(self.dataset_name)
    
    def load_paraphrased_dataset(self):
        """Load paraphrased dataset from CSV"""
        print(f"Loading paraphrased data from: {self.paraphrased_csv_path}")
        df = pd.read_csv(self.paraphrased_csv_path)
        
        # Convert to Dataset format
        datasets = {}
        for split in df['split'].unique():
            split_df = df[df['split'] == split]
            datasets[split] = Dataset.from_dict({
                'text': split_df['paraphrased_text'].tolist(),
                'labels': split_df['original_label'].tolist()
            })
        
        return datasets
    
    def create_temporary_datasets(self):
        """Create temporary dataset files for experiments"""
        self.temp_dir = tempfile.mkdtemp(prefix="paraphrase_exp_")
        print(f"Creating temporary datasets in: {self.temp_dir}")
        
        # Load datasets
        original = self.load_original_dataset()
        paraphrased = self.load_paraphrased_dataset()
        
        # Create experiment dataset combinations
        experiments = {
            'exp1': {
                'train': paraphrased['train'],
                'test': original['test'],
                'name': 'paraphrased_train_original_test'
            },
            'exp2': {
                'train': paraphrased['train'],
                'test': paraphrased['test'],
                'name': 'paraphrased_train_paraphrased_test'
            },
            'exp3_baseline': {
                'train': original['train'],
                'test': original['test'],
                'name': 'original_train_original_test'
            }
        }
        
        dataset_paths = {}
        
        for exp_name, exp_data in experiments.items():
            exp_dir = Path(self.temp_dir) / exp_name
            exp_dir.mkdir(exist_ok=True)
            
            # Save train and test splits
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
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")


def create_hydra_config(exp_name, dataset_paths, output_dir, seed=42):
    """Create Hydra config for an experiment"""
    config = {
        'defaults': ['_self_'],
        'hydra': {
            'job': {'chdir': False},
            'run': {'dir': str(Path(output_dir) / exp_name)},
        },
        'params': {
            'dataset': 'custom',  # We'll use custom loading
            'seed': seed,
            'per_device_batch_size': 8,
            'freeze': 0,
            'pretrained': True,
            'use_spurious': False,  # No spurious correlation for these experiments
            'backbone': 'apple/OpenELM-450M',
            'lora_rank': 0,
            'training_steps': 500,
            'batch_size': 64,
            'pretrained_tokenizer': None,
            'weight_decay': 1e-5,
            'learning_rate': 2e-5,
            'dropout': 0,
            'mixup': 0,
            'vocab_size': None,
            'max_length': 512,
            'label_smoothing': 0,
            'from_gcs': 'none',
            'eval_steps': 100,
            'mixture': 0,
            'lora0': 0,
            'superlinear': 'none',
            'scaling_gamma': 0,
            'use_dora': False,
            'spurious_location': 'random',
            'spurious_proportion': 1,
            'spurious_token_proportion': 0.1,
            'spurious_label': 1,
            'total_parameters': 0,
            'training_parameters': 0,
            'spurious_test_label': 1,
            'spurious_test_proportion': 1,
            'spurious_test_token_proportion': 0.1,
            'spurious_test_location': 'random',
            'spurious_type': 'date',
            'use_list_dataset': False,
            'list_dataset_path': False,
            'with_replacement': True,
            'date_range': [1900, 2100],
            'getting_num_preds': 0,
            # Custom paths for our experiment
            'custom_train_path': dataset_paths['train_path'],
            'custom_test_path': dataset_paths['test_path'],
        }
    }
    return config


def run_single_experiment(exp_name, dataset_paths, output_dir, seed=42):
    """Run a single experiment using the modified supervised_finetuning.py"""
    print(f"\n{'='*50}")
    print(f"Running {exp_name}: {dataset_paths['name']}")
    print(f"{'='*50}")
    
    # Create config file
    config = create_hydra_config(exp_name, dataset_paths, output_dir, seed)
    
    config_dir = Path(output_dir) / exp_name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "hydra.yaml"
    
    # Write config to YAML format
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run the experiment
    cmd = [
        sys.executable,
        "/home/ubuntu/Spurious_corr_paraphrase/paraphrase_finetuning_modified.py",
        f"--config-path={config_dir}",
        f"--config-name=hydra"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd="/home/ubuntu/Spurious_corr_paraphrase"
        )
        
        print("Experiment completed successfully!")
        return True, result.stdout, result.stderr
        
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, e.stdout, e.stderr


def extract_metrics_from_output(stdout, stderr):
    """Extract metrics from the training output"""
    metrics = {}
    
    # Look for evaluation metrics in the output
    lines = (stdout + stderr).split('\n')
    
    for line in lines:
        if 'eval_accuracy' in line:
            try:
                # Extract accuracy value
                parts = line.split('eval_accuracy')
                if len(parts) > 1:
                    value_part = parts[1].strip()
                    value = float(value_part.split()[0].replace(':', '').replace(',', ''))
                    metrics['accuracy'] = value
            except:
                pass
        
        if 'eval_F1' in line:
            try:
                parts = line.split('eval_F1')
                if len(parts) > 1:
                    value_part = parts[1].strip()
                    value = float(value_part.split()[0].replace(':', '').replace(',', ''))
                    metrics['f1'] = value
            except:
                pass
    
    return metrics


def main():
    """Main function to run all experiments"""
    print("Starting Paraphrase Finetuning Experiments")
    print("=" * 60)
    
    # Configuration
    dataset_name = "rotten_tomatoes"
    output_dir = "/home/ubuntu/Spurious_corr_paraphrase/experiment_outputs"
    seed = 42
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset creator
    dataset_creator = ParaphraseDatasetCreator(dataset_name)
    
    try:
        # Create temporary datasets
        print("Creating temporary datasets...")
        dataset_paths = dataset_creator.create_temporary_datasets()
        
        # Results storage
        all_results = {}
        
        # Run experiments
        experiments = [
            ('experiment_1', dataset_paths['exp1'], 'Train on Paraphrased → Eval on Original Test'),
            ('experiment_2', dataset_paths['exp2'], 'Train on Paraphrased → Eval on Paraphrased Test'),
            ('experiment_3_baseline', dataset_paths['exp3_baseline'], 'Train on Original → Eval on Original Test (BASELINE)')
        ]
        
        for exp_id, exp_paths, exp_description in experiments:
            print(f"\n--- {exp_description} ---")
            
            success, stdout, stderr = run_single_experiment(exp_id, exp_paths, output_dir, seed)
            
            if success:
                metrics = extract_metrics_from_output(stdout, stderr)
                all_results[exp_id] = {
                    'description': exp_description,
                    'metrics': metrics,
                    'success': True
                }
                print(f"✓ {exp_description} completed successfully")
                if metrics:
                    print(f"  Metrics: {metrics}")
            else:
                all_results[exp_id] = {
                    'description': exp_description,
                    'success': False,
                    'error': stderr
                }
                print(f"✗ {exp_description} failed")
        
        # Save final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'seed': seed,
            'experiments': all_results
        }
        
        results_file = Path(output_dir) / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 60)
        
        for exp_id, result in all_results.items():
            print(f"\n{result['description']}:")
            if result['success'] and result.get('metrics'):
                metrics = result['metrics']
                if 'accuracy' in metrics:
                    print(f"  Accuracy: {metrics['accuracy']:.4f}")
                if 'f1' in metrics:
                    print(f"  F1 Score: {metrics['f1']:.4f}")
            else:
                print("  Status: Failed or no metrics available")
        
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error running experiments: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup temporary files
        dataset_creator.cleanup()


if __name__ == "__main__":
    main()
