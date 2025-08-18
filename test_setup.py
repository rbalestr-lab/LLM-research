#!/usr/bin/env python3
"""
Test setup for paraphrase experiments

This script tests if all required components are available.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Set up environment
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/opt/dlami/nvme/hf_cache/models"

# Add paths
sys.path.append('/home/ubuntu/research_workspace/LLM-research')

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not available")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not available")
        return False
    
    try:
        import llm_research
        print("✓ LLM Research framework available")
    except ImportError:
        print("✗ LLM Research framework not available")
        return False
    
    try:
        from datasets import Dataset
        print("✓ Datasets library available")
    except ImportError:
        print("✗ Datasets library not available")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn available")
    except ImportError:
        print("✗ Scikit-learn not available")
        return False
    
    return True


def test_data_availability():
    """Test if required data is available"""
    print("\nTesting data availability...")
    
    try:
        import llm_research
        data = llm_research.data.from_name("rotten_tomatoes")
        print(f"✓ Original rotten_tomatoes dataset: train={len(data['train'])}, test={len(data['test'])}")
    except Exception as e:
        print(f"✗ Failed to load original dataset: {e}")
        return False
    
    # Check paraphrased data
    base_path = Path("/home/ubuntu/Spurious_corr_paraphrase/pr_dataset/rotten_tomatoes")
    csv_files = list(base_path.glob("**/*.csv"))
    
    if csv_files:
        csv_file = csv_files[0]
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Paraphrased data available: {len(df)} examples in {csv_file}")
            
            # Check splits
            splits = df['split'].value_counts()
            for split, count in splits.items():
                print(f"  {split}: {count} examples")
                
        except Exception as e:
            print(f"✗ Failed to load paraphrased data: {e}")
            return False
    else:
        print("⚠ No paraphrased data found - will be generated when needed")
    
    return True


def test_gpu_availability():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False
    
    return True


def test_cache_directory():
    """Test cache directory setup"""
    print("\nTesting cache directories...")
    
    cache_dirs = [
        "/opt/dlami/nvme/hf_cache",
        "/opt/dlami/nvme/hf_cache/datasets",
        "/opt/dlami/nvme/hf_cache/models"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"✓ Cache directory exists: {cache_dir}")
        else:
            print(f"⚠ Cache directory missing: {cache_dir}")
            try:
                os.makedirs(cache_dir, exist_ok=True)
                print(f"✓ Created cache directory: {cache_dir}")
            except Exception as e:
                print(f"✗ Failed to create cache directory: {e}")
                return False
    
    return True


def main():
    """Run all tests"""
    print("Paraphrase Experiments Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Availability", test_data_availability),
        ("GPU Availability", test_gpu_availability),
        ("Cache Directories", test_cache_directory)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Setup is ready for experiments.")
        print("\nTo run experiments:")
        print("  python run_complete_experiments.py")
    else:
        print("✗ Some tests failed. Please fix issues before running experiments.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
