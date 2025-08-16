#!/usr/bin/env python3
"""
Command-line interface for running paraphrasing with specific datasets and models.

Usage:
    python run_paraphrasing.py --dataset rotten_tomatoes --model meta-llama/Meta-Llama-3-8B
    python run_paraphrasing.py --dataset sst2 --model openai/gpt-oss-20b
    python run_paraphrasing.py --list-datasets
    python run_paraphrasing.py --list-models
"""

import argparse
import sys
import os
import traceback
import torch
import gc

# Add research workspace to path
sys.path.append('/home/ubuntu/research_workspace/LLM-research')

# Set environment variables
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/opt/dlami/nvme/hf_cache/models"

from llm_research.data import from_name as data_from_name, NAMES as DATASET_NAMES
from llm_research import MODELS
from paraphraser import LLMInterface, process_dataset_paraphrasing, process_dataset_paraphrasing_concurrent, save_results_to_csv

# Available datasets and models
AVAILABLE_DATASETS = [
    "rotten_tomatoes",
    "sst2", 
    "yelp_review_full",
    "imdb",
    "emotion",
    "polarity",
    "financial_classification"
]

AVAILABLE_MODELS = [
    model for model in MODELS 
    if any(size in model for size in ["1.5B", "3B", "7B", "8B", "70B", "24B", "20b", "120b"]) 
    or "phi-2" in model
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run paraphrasing with specified dataset and model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset rotten_tomatoes --model meta-llama/Meta-Llama-3-8B
  %(prog)s --dataset sst2 --model openai/gpt-oss-20b --batch-size 512
  %(prog)s --dataset emotion --run-all-models --skip-errors
  %(prog)s --model microsoft/phi-2 --run-all-datasets --skip-errors
  %(prog)s --run-everything --skip-errors --max-examples 100
  %(prog)s --list-datasets
  %(prog)s --list-models
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use for paraphrasing"
    )
    
    parser.add_argument(
        "--model", "--llm-model",
        type=str,
        help="LLM model to use for paraphrasing"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for processing (default: 1024)"
    )
    
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to process per split"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pr_dataset",
        help="Output directory for results (default: pr_dataset)"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )
    
    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Show GPU information"
    )
    
    parser.add_argument(
        "--run-all-models",
        action="store_true",
        help="Run paraphrasing with all available models for the specified dataset"
    )
    
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue with next model if one fails (useful with --run-all-models)"
    )
    
    parser.add_argument(
        "--run-all-datasets",
        action="store_true",
        help="Run paraphrasing with all available datasets for the specified model"
    )
    
    parser.add_argument(
        "--run-everything",
        action="store_true",
        help="Run ALL datasets with ALL models (comprehensive evaluation)"
    )
    
    return parser.parse_args()

def list_datasets():
    """List all available datasets"""
    print("Available Datasets:")
    print("=" * 50)
    for i, dataset in enumerate(AVAILABLE_DATASETS, 1):
        print(f"{i:2d}. {dataset}")
    print(f"\nTotal: {len(AVAILABLE_DATASETS)} datasets")

def list_models():
    """List all available models"""
    print("Available Models:")
    print("=" * 50)
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        # Estimate model size for display
        size = "Unknown"
        if "120b" in model.lower():
            size = "~117B params, ~80GB VRAM"
        elif "20b" in model.lower():
            size = "~21B params, ~16GB VRAM"
        elif "70b" in model.lower():
            size = "~70B params, ~40GB VRAM"
        elif "8b" in model.lower():
            size = "~8B params, ~8GB VRAM"
        elif "7b" in model.lower():
            size = "~7B params, ~7GB VRAM"
        elif "3b" in model.lower():
            size = "~3B params, ~3GB VRAM"
        elif "1.5b" in model.lower():
            size = "~1.5B params, ~2GB VRAM"
        elif "phi-2" in model.lower():
            size = "~2.7B params, ~3GB VRAM"
            
        print(f"{i:2d}. {model}")
        if size != "Unknown":
            print(f"    └─ {size}")
    print(f"\nTotal: {len(AVAILABLE_MODELS)} models")

def show_gpu_info():
    """Show GPU information"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU Information:")
        print("=" * 50)
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {memory_gb:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("No CUDA GPUs available")

def get_model_family(model_name):
    """Extract model family name for organized folder structure"""
    if "openai" in model_name.lower():
        return "openai"
    elif "meta-llama" in model_name.lower():
        return "meta-llama"
    elif "mistralai" in model_name.lower():
        return "mistralai"
    elif "microsoft" in model_name.lower():
        return "microsoft"
    elif "apple" in model_name.lower():
        return "apple"
    elif "qwen" in model_name.lower():
        return "qwen"
    elif "google" in model_name.lower():
        return "google"
    else:
        return "other"

def save_results_organized(results, dataset_name, model_name, output_dir="pr_dataset"):
    """Save results with organized folder structure: dataset/model_family/model_name.csv"""
    import pandas as pd
    
    all_data = []
    for split_name, split_results in results.items():
        for result in split_results:
            all_data.append({
                'split': split_name,
                'original_text': result['original_text'],
                'original_label': result['original_label'],
                'paraphrased_text': result['paraphrased_text']
            })
    
    # Create organized folder structure
    dataset_clean = dataset_name.replace("/", "_").replace("-", "_")
    model_family = get_model_family(model_name)
    model_clean = model_name.replace("/", "_").replace("-", "_")
    
    # Create directory structure: pr_dataset/dataset_name/model_family/
    output_path = os.path.join(output_dir, dataset_clean, model_family)
    os.makedirs(output_path, exist_ok=True)
    
    # Save file as: model_name.csv
    csv_filename = f"{model_clean}.csv"
    full_filename = os.path.join(output_path, csv_filename)
    
    df = pd.DataFrame(all_data)
    df.to_csv(full_filename, index=False, encoding='utf-8')
    print(f"Results saved to: {full_filename} ({len(df)} rows)")
    
    return df, full_filename

def load_dataset(dataset_name, max_examples=None):
    """Load and optionally limit dataset"""
    print(f"Loading dataset: {dataset_name}")
    dataset = data_from_name(dataset_name)
    
    if max_examples:
        print(f"Limiting to {max_examples} examples per split")
        limited_dataset = {}
        for split_name, split_data in dataset.items():
            if len(split_data) > max_examples:
                limited_dataset[split_name] = split_data.select(range(max_examples))
            else:
                limited_dataset[split_name] = split_data
        dataset = limited_dataset
    
    return dataset

def run_all_models(args):
    """Run paraphrasing with all available models for a dataset"""
    print(f"🚀 Running All Models for Dataset: {args.dataset}")
    print("=" * 60)
    print(f"Total models to process: {len(AVAILABLE_MODELS)}")
    print(f"Skip errors: {'Yes' if args.skip_errors else 'No'}")
    print()
    
    # Load dataset once
    dataset = load_dataset(args.dataset, args.max_examples)
    
    successful_runs = []
    failed_runs = []
    
    for i, model_name in enumerate(AVAILABLE_MODELS, 1):
        print(f"\n[{i}/{len(AVAILABLE_MODELS)}] Processing: {model_name}")
        print("-" * 50)
        
        try:
            # Initialize model
            print(f"Loading model: {model_name}")
            llm = LLMInterface(model_name=model_name)
            
            # Run paraphrasing
            print("Starting paraphrasing process...")
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            if num_gpus >= 2:
                results = process_dataset_paraphrasing_concurrent(
                    llm, dataset, batch_size=args.batch_size
                )
            else:
                results = process_dataset_paraphrasing(
                    llm, dataset, batch_size=args.batch_size
                )
            
            # Save results with organized structure
            if results:
                total_processed = sum(len(split_results) for split_results in results.values())
                print(f"Total processed: {total_processed}")
                
                df, output_file = save_results_organized(results, args.dataset, model_name, args.output_dir)
                print(f"✅ Successfully saved {len(df)} paraphrased examples")
                successful_runs.append((model_name, output_file, len(df)))
            else:
                print(f"❌ No results for {model_name}")
                failed_runs.append((model_name, "No results generated"))
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error with {model_name}: {error_msg}")
            failed_runs.append((model_name, error_msg))
            
            if not args.skip_errors:
                print("Stopping due to error. Use --skip-errors to continue with next model.")
                break
        
        finally:
            # Cleanup after each model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(failed_runs)}")
    
    if successful_runs:
        print(f"\n✅ Successful runs:")
        for model, file_path, count in successful_runs:
            print(f"  - {model}: {count} examples → {file_path}")
    
    if failed_runs:
        print(f"\n❌ Failed runs:")
        for model, error in failed_runs:
            print(f"  - {model}: {error}")

def run_all_datasets(args):
    """Run paraphrasing with all available datasets for a model"""
    print(f"🚀 Running All Datasets for Model: {args.model}")
    print("=" * 60)
    print(f"Total datasets to process: {len(AVAILABLE_DATASETS)}")
    print(f"Skip errors: {'Yes' if args.skip_errors else 'No'}")
    print()
    
    successful_runs = []
    failed_runs = []
    
    for i, dataset_name in enumerate(AVAILABLE_DATASETS, 1):
        print(f"\n[{i}/{len(AVAILABLE_DATASETS)}] Processing dataset: {dataset_name}")
        print("-" * 50)
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, args.max_examples)
            
            # Initialize model
            print(f"Loading model: {args.model}")
            llm = LLMInterface(model_name=args.model)
            
            # Run paraphrasing
            print("Starting paraphrasing process...")
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            if num_gpus >= 2:
                results = process_dataset_paraphrasing_concurrent(
                    llm, dataset, batch_size=args.batch_size
                )
            else:
                results = process_dataset_paraphrasing(
                    llm, dataset, batch_size=args.batch_size
                )
            
            # Save results with organized structure
            if results:
                total_processed = sum(len(split_results) for split_results in results.values())
                print(f"Total processed: {total_processed}")
                
                df, output_file = save_results_organized(results, dataset_name, args.model, args.output_dir)
                print(f"✅ Successfully saved {len(df)} paraphrased examples")
                successful_runs.append((dataset_name, output_file, len(df)))
            else:
                print(f"❌ No results for {dataset_name}")
                failed_runs.append((dataset_name, "No results generated"))
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error with {dataset_name}: {error_msg}")
            failed_runs.append((dataset_name, error_msg))
            
            if not args.skip_errors:
                print("Stopping due to error. Use --skip-errors to continue with next dataset.")
                break
        
        finally:
            # Cleanup after each dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(failed_runs)}")
    
    if successful_runs:
        print(f"\n✅ Successful runs:")
        for dataset, file_path, count in successful_runs:
            print(f"  - {dataset}: {count} examples → {file_path}")
    
    if failed_runs:
        print(f"\n❌ Failed runs:")
        for dataset, error in failed_runs:
            print(f"  - {dataset}: {error}")

def run_everything(args):
    """Run ALL datasets with ALL models (comprehensive evaluation)"""
    print(f"🚀 COMPREHENSIVE EVALUATION: All Datasets × All Models")
    print("=" * 70)
    print(f"Total combinations: {len(AVAILABLE_DATASETS)} datasets × {len(AVAILABLE_MODELS)} models = {len(AVAILABLE_DATASETS) * len(AVAILABLE_MODELS)}")
    print(f"Skip errors: {'Yes' if args.skip_errors else 'No'}")
    if args.max_examples:
        print(f"Max examples per dataset split: {args.max_examples}")
    print()
    
    total_successful = 0
    total_failed = 0
    all_results = {}
    
    for d_idx, dataset_name in enumerate(AVAILABLE_DATASETS, 1):
        print(f"\n{'='*70}")
        print(f"DATASET {d_idx}/{len(AVAILABLE_DATASETS)}: {dataset_name}")
        print(f"{'='*70}")
        
        # Load dataset once for all models
        try:
            dataset = load_dataset(dataset_name, args.max_examples)
        except Exception as e:
            print(f"❌ Failed to load dataset {dataset_name}: {e}")
            if not args.skip_errors:
                break
            continue
        
        dataset_results = {}
        
        for m_idx, model_name in enumerate(AVAILABLE_MODELS, 1):
            print(f"\n[{d_idx}.{m_idx}] {dataset_name} × {model_name}")
            print("-" * 50)
            
            try:
                # Initialize model
                print(f"Loading model: {model_name}")
                llm = LLMInterface(model_name=model_name)
                
                # Run paraphrasing
                print("Starting paraphrasing process...")
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                
                if num_gpus >= 2:
                    results = process_dataset_paraphrasing_concurrent(
                        llm, dataset, batch_size=args.batch_size
                    )
                else:
                    results = process_dataset_paraphrasing(
                        llm, dataset, batch_size=args.batch_size
                    )
                
                # Save results
                if results:
                    total_processed = sum(len(split_results) for split_results in results.values())
                    df, output_file = save_results_organized(results, dataset_name, model_name, args.output_dir)
                    print(f"✅ Success: {len(df)} examples → {output_file}")
                    dataset_results[model_name] = {"success": True, "count": len(df), "file": output_file}
                    total_successful += 1
                else:
                    print(f"❌ No results generated")
                    dataset_results[model_name] = {"success": False, "error": "No results generated"}
                    total_failed += 1
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error: {error_msg}")
                dataset_results[model_name] = {"success": False, "error": error_msg}
                total_failed += 1
                
                if not args.skip_errors:
                    print("Stopping due to error. Use --skip-errors to continue.")
                    return
            
            finally:
                # Cleanup after each model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
        
        all_results[dataset_name] = dataset_results
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total combinations attempted: {total_successful + total_failed}")
    print(f"Successful runs: {total_successful}")
    print(f"Failed runs: {total_failed}")
    print(f"Success rate: {total_successful/(total_successful + total_failed)*100:.1f}%" if (total_successful + total_failed) > 0 else "0%")
    
    # Dataset-wise summary
    print(f"\n📊 Results by Dataset:")
    for dataset, results in all_results.items():
        successful = sum(1 for r in results.values() if r["success"])
        total = len(results)
        print(f"  {dataset}: {successful}/{total} models successful")
    
    # Model-wise summary
    print(f"\n🤖 Results by Model:")
    model_stats = {}
    for dataset_results in all_results.values():
        for model, result in dataset_results.items():
            if model not in model_stats:
                model_stats[model] = {"success": 0, "total": 0}
            model_stats[model]["total"] += 1
            if result["success"]:
                model_stats[model]["success"] += 1
    
    for model, stats in model_stats.items():
        success_rate = stats["success"]/stats["total"]*100 if stats["total"] > 0 else 0
        print(f"  {model}: {stats['success']}/{stats['total']} datasets ({success_rate:.1f}%)")

def run_paraphrasing(args):
    """Run the paraphrasing process"""
    print(f"🚀 Starting Paraphrasing")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    if args.max_examples:
        print(f"Max Examples: {args.max_examples}")
    print()
    
    try:
        # Load dataset
        dataset = load_dataset(args.dataset, args.max_examples)
        
        # Initialize model
        print(f"Loading model: {args.model}")
        llm = LLMInterface(model_name=args.model)
        
        # Run paraphrasing
        print("Starting paraphrasing process...")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus >= 2:
            print(f"Using concurrent processing with {num_gpus} GPUs")
            results = process_dataset_paraphrasing_concurrent(
                llm, dataset, batch_size=args.batch_size
            )
        else:
            print("Using single-GPU processing")
            results = process_dataset_paraphrasing(
                llm, dataset, batch_size=args.batch_size
            )
        
        # Save results with organized structure
        if results:
            total_processed = sum(len(split_results) for split_results in results.values())
            print(f"Total processed: {total_processed}")
            
            df, output_file = save_results_organized(results, args.dataset, args.model, args.output_dir)
            print(f"✅ Successfully saved {len(df)} paraphrased examples")
            print(f"Output file: {output_file}")
        else:
            print("❌ No results to save - processing failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("Cleanup completed")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Handle list commands
    if args.list_datasets:
        list_datasets()
        return
    
    if args.list_models:
        list_models()
        return
        
    if args.gpu_info:
        show_gpu_info()
        return
    
    # Handle comprehensive evaluation (run everything)
    if args.run_everything:
        # Run everything doesn't need validation
        run_everything(args)
        return
    
    # Handle run-all-models command
    if args.run_all_models:
        if not args.dataset:
            print("❌ Error: --dataset is required when using --run-all-models")
            print("Use --list-datasets to see available options")
            sys.exit(1)
        
        # Validate dataset for run-all-models
        if args.dataset not in AVAILABLE_DATASETS:
            print(f"❌ Error: Dataset '{args.dataset}' not available")
            print("Available datasets:")
            for dataset in AVAILABLE_DATASETS:
                print(f"  - {dataset}")
            sys.exit(1)
        
        # Run all models
        run_all_models(args)
        return
    
    # Handle run-all-datasets command
    if args.run_all_datasets:
        if not args.model:
            print("❌ Error: --model is required when using --run-all-datasets")
            print("Use --list-models to see available options")
            sys.exit(1)
        
        # Validate model for run-all-datasets
        if args.model not in AVAILABLE_MODELS:
            print(f"❌ Error: Model '{args.model}' not available")
            print("Available models:")
            for model in AVAILABLE_MODELS[:5]:
                print(f"  - {model}")
            print(f"  ... and {len(AVAILABLE_MODELS)-5} more (use --list-models to see all)")
            sys.exit(1)
        
        # Run all datasets
        run_all_datasets(args)
        return
    
    # Validate required arguments for single model run
    if not args.dataset or not args.model:
        print("❌ Error: Both --dataset and --model are required")
        print("Use --list-datasets and --list-models to see available options")
        print("Or use --run-all-models to run with all models")
        sys.exit(1)
    
    # Validate dataset
    if args.dataset not in AVAILABLE_DATASETS:
        print(f"❌ Error: Dataset '{args.dataset}' not available")
        print("Available datasets:")
        for dataset in AVAILABLE_DATASETS:
            print(f"  - {dataset}")
        sys.exit(1)
    
    # Validate model
    if args.model not in AVAILABLE_MODELS:
        print(f"❌ Error: Model '{args.model}' not available")
        print("Available models:")
        for model in AVAILABLE_MODELS[:5]:  # Show first 5
            print(f"  - {model}")
        print(f"  ... and {len(AVAILABLE_MODELS)-5} more (use --list-models to see all)")
        sys.exit(1)
    
    # Run paraphrasing
    run_paraphrasing(args)

if __name__ == "__main__":
    main()
