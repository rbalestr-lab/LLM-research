#!/usr/bin/env python3
"""
Script to run paraphraser.py for all models in LARGE_MODELS list
Saves results in organized folder structure: pr_dataset/dataset_name/paraphrase_model_name.csv
Models are cached in memory to avoid reloading
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
    LARGE_MODELS,
    DATASETS,
    BATCH_SIZE
)

MODEL_CACHE = {}

def setup_output_directory(dataset_name):
    """
    Create the output directory structure: pr_dataset/dataset_name
    """
    dataset_clean = dataset_name.replace("/", "_").replace("-", "_")
    output_dir = Path("pr_dataset") / dataset_clean
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_cached_model(model_name):
    """
    Get model from cache or load it if not cached
    """
    if model_name not in MODEL_CACHE:
        print(f"Loading model into cache: {model_name}")
        MODEL_CACHE[model_name] = LLMInterface(model_name=model_name)
        print(f" Model cached: {model_name}")
    else:
        print(f" Using cached model: {model_name}")
    return MODEL_CACHE[model_name]

def clear_model_cache():
    """
    Clear all cached models if needed (optional function)
    """
    global MODEL_CACHE
    for model_name in list(MODEL_CACHE.keys()):
        del MODEL_CACHE[model_name]
    MODEL_CACHE.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(" Model cache cleared")

def get_cache_status():
    """
    Get information about cached models
    """
    if not MODEL_CACHE:
        return "No models cached"
    cached_models = list(MODEL_CACHE.keys())
    return f"Cached models ({len(cached_models)}): {', '.join(cached_models)}"

def run_paraphrasing_for_model(model_name, dataset_name, dataset, output_dir):
    """
    Run paraphrasing for a single model and dataset combination, save results
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {dataset_name} with {model_name}")
    print(f"{'='*80}")
    
    try:
        llm = get_cached_model(model_name)
        print(f"Using batch size: {BATCH_SIZE}")
        results = process_dataset_paraphrasing(llm, dataset, batch_size=BATCH_SIZE)
        
        if results:
            total_processed = sum(len(split_results) for split_results in results.values())
            print(f"Total processed: {total_processed}")
            model_clean = model_name.replace("/", "_").replace("-", "_")
            filename = output_dir / f"{model_clean}.csv"
            df = save_results_to_csv(results, dataset_name, model_name, filename=filename.name)
            print(f" Successfully saved {len(df)} examples to: {filename}")
            return True
        else:
            print(f" No results generated for {dataset_name} with {model_name}")
            return False
            
    except Exception as e:
        print(f" Error processing {dataset_name} with {model_name}: {e}")
        traceback.print_exc()
        return False
    
    finally:
        print(f"Model {model_name} processing complete (model remains cached)")
        print(f"Cache status: {get_cache_status()}")

def main():
    """
    Main function to process all models for all datasets
    """
    print(f"Datasets to process: {len(DATASETS)}")
    for i, dataset in enumerate(DATASETS, 1):
        print(f"  {i}. {dataset}")
    print(f"Models to process: {len(LARGE_MODELS)}")
    for i, model in enumerate(LARGE_MODELS, 1):
        print(f"  {i}. {model}")
    print("="*80)
    
    print("Loading datasets...")
    all_datasets = load_datasets()
    print(f"Successfully loaded {len(all_datasets)} datasets: {list(all_datasets.keys())}")
    
    successful_combinations = []
    failed_combinations = []
    
    total_combinations = len(DATASETS) * len(LARGE_MODELS)
    
    for dataset_name in DATASETS:
        if dataset_name not in all_datasets:
            print(f"  Dataset {dataset_name} not loaded, skipping...")
            continue
            
        dataset = all_datasets[dataset_name]
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"Dataset splits: {list(dataset.keys())}")
        print(f"{'='*60}")
        
        for model_name in LARGE_MODELS:
            print(f"Dataset: {dataset_name} | Model: {model_name}")

            output_dir = setup_output_directory(dataset_name)
            model_clean = model_name.replace("/", "_").replace("-", "_")
            output_file = output_dir / f"{model_clean}.csv"
            
            if output_file.exists():
                print(f" File already exists: {output_file}")
                successful_combinations.append((dataset_name, model_name))
                continue
            
            success = run_paraphrasing_for_model(model_name, dataset_name, dataset, output_dir)
            
            if success:
                successful_combinations.append((dataset_name, model_name))
            else:
                failed_combinations.append((dataset_name, model_name))
            
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Successful: {len(successful_combinations)}")
    print(f"Failed: {len(failed_combinations)}")
    print(f" {get_cache_status()}")
    
    if successful_combinations:
        print(f"\n Successful:")
        for dataset_name, model_name in successful_combinations:
            dataset_clean = dataset_name.replace("/", "_").replace("-", "_")
            model_clean = model_name.replace("/", "_").replace("-", "_")
            output_path = f"pr_dataset/{dataset_clean}/{model_clean}.csv"
            print(f"  - {dataset_name} + {model_name} → {output_path}")
    
    if failed_combinations:
        print(f"\n Failed:")
        for dataset_name, model_name in failed_combinations:
            print(f"  - {dataset_name} + {model_name}")

if __name__ == "__main__":
    main() 