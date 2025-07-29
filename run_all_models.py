#!/usr/bin/env python3
"""
Script to run paraphraser.py for all models in LARGE_MODELS list
Saves results in organized folder structure: pr_dataset/dataset_name/paraphrase_model_name.csv
"""

import os
import sys
import traceback
import gc
import torch
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paraphraser import (
    LLMInterface, 
    load_datasets,
    process_dataset_paraphrasing,
    save_results_to_csv,
    process_dataset_paraphrasing_concurrent,
    LARGE_MODELS,
    DATASETS,
    BATCH_SIZE
)

MODEL_CACHE = {}

def setup_output_directory(dataset_name, model_name):
    dataset_clean = dataset_name.replace("/", "_").replace("-", "_")
    model_clean = model_name.replace("/", "_").replace("-", "_")
    output_dir = Path("pr_dataset") / dataset_clean / model_clean
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_cached_model(model_name):
    if model_name not in MODEL_CACHE:
        print(f"Loading model: {model_name}")
        MODEL_CACHE[model_name] = LLMInterface(model_name=model_name)
    return MODEL_CACHE[model_name]

def run_paraphrasing_for_model(model_name, dataset_name, dataset, output_dir):
    print(f"\nProcessing: {dataset_name} with {model_name}")
    
    try:
        llm = get_cached_model(model_name)
        batch_size = BATCH_SIZE
        
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus >= 2:
            results = process_dataset_paraphrasing_concurrent(llm, dataset, batch_size=batch_size)
        else:
            results = process_dataset_paraphrasing(llm, dataset, batch_size=batch_size)
        
        if results:
            total_processed = sum(len(split_results) for split_results in results.values())
            dataset_clean = dataset_name.replace("/", "_").replace("-", "_")
            filename = output_dir / f"paraphrased_{dataset_clean}.csv"
            df = save_results_to_csv(results, dataset_name, model_name, filename=filename.name)
            print(f"Saved {len(df)} examples to: {filename}")
            return True
        else:
            print(f"No results generated for {dataset_name} with {model_name}")
            return False
            
    except Exception as e:
        print(f"Error processing {dataset_name} with {model_name}: {e}")
        traceback.print_exc()
        return False

def main():
    print("BULK PARAPHRASING FOR ALL MODELS AND DATASETS")
    print(f"Datasets: {len(DATASETS)}, Models: {len(LARGE_MODELS)}")
    print(f"Total combinations: {len(DATASETS) * len(LARGE_MODELS)}")
    
    all_datasets = load_datasets()
    print(f"Loaded {len(all_datasets)} datasets")
    
    successful_combinations = []
    failed_combinations = []
    combination_count = 0
    total_combinations = len(DATASETS) * len(LARGE_MODELS)
    
    for dataset_name in DATASETS:
        if dataset_name not in all_datasets:
            print(f"Dataset {dataset_name} not loaded, skipping...")
            continue
            
        dataset = all_datasets[dataset_name]
        
        for model_name in LARGE_MODELS:
            combination_count += 1
            print(f"\nProcessing {combination_count}/{total_combinations}: {dataset_name} | {model_name}")
            
            output_dir = setup_output_directory(dataset_name, model_name)
            dataset_clean = dataset_name.replace("/", "_").replace("-", "_")
            output_file = output_dir / f"paraphrased_{dataset_clean}.csv"
            
            if output_file.exists():
                print(f"File exists, skipping: {output_file}")
                successful_combinations.append((dataset_name, model_name))
                continue
            
            success = run_paraphrasing_for_model(model_name, dataset_name, dataset, output_dir)
            
            if success:
                successful_combinations.append((dataset_name, model_name))
            else:
                failed_combinations.append((dataset_name, model_name))
    
    print(f"\nFINAL SUMMARY")
    print(f"Total: {total_combinations}, Successful: {len(successful_combinations)}, Failed: {len(failed_combinations)}")
    
    if failed_combinations:
        print("Failed combinations:")
        for dataset_name, model_name in failed_combinations:
            print(f"  - {dataset_name} + {model_name}")

if __name__ == "__main__":
    main() 