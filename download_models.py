#!/usr/bin/env python3
"""
Download all specified models to /opt/dlami/nvme/hf_cache/models
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

# Set cache directory
CACHE_DIR = "/opt/dlami/nvme/hf_cache/models"
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# Load environment for HF token
try:
    load_dotenv("/home/ubuntu/Spurious_corr_paraphrase/.env")
except:
    pass

# Your specified models
LARGE_MODELS = [
     "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-1.5B",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-Small-24B-Base-2501",
    "google/gemma-7b",
    "google/gemma-2b",
    "microsoft/phi-2",
    # "meta-llama/Llama-3.2-1B",
    # "microsoft/DialoGPT-small",
    # "distilbert-base-uncased",
    # "Snowflake/snowflake-arctic-embed-l"
]

def check_model_cached(model_name):
    """Check if model is already cached"""
    model_dir = model_name.replace("/", "--")
    cache_path = os.path.join(CACHE_DIR, f"models--{model_dir}")
    return os.path.exists(cache_path)

def download_model(model_name):
    """Download model to cache"""
    print(f"📥 Downloading {model_name}...")
    
    try:
        # Get HF token if available
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        
        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=True)
                print("✅ Authenticated with HuggingFace")
            except Exception as e:
                print(f"⚠️ Authentication failed: {e}")
        
        # Download tokenizer (lightweight)
        print(f"  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            token=hf_token,
            trust_remote_code=True
        )
        print(f"  ✅ Tokenizer downloaded")
        
        # Download model (heavier)
        print(f"  Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            token=hf_token,
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None  # Don't load to GPU, just download
        )
        print(f"  ✅ Model weights downloaded")
        
        # Clear from memory
        del model, tokenizer
        
        print(f"✅ {model_name} successfully cached to {CACHE_DIR}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def main():
    print("🚀 Downloading Models to Cache Directory")
    print("="*60)
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"Models to download: {len(LARGE_MODELS)}")
    print("="*60)
    
    # Check what's already cached
    cached_models = []
    missing_models = []
    
    for model_name in LARGE_MODELS:
        if check_model_cached(model_name):
            cached_models.append(model_name)
            print(f"✅ {model_name} - Already cached")
        else:
            missing_models.append(model_name)
            print(f"❌ {model_name} - Missing")
    
    print(f"\n📊 Summary:")
    print(f"  Already cached: {len(cached_models)}")
    print(f"  Need to download: {len(missing_models)}")
    
    if not missing_models:
        print(f"\n🎉 All models already cached!")
        return
    
    print(f"\n📥 Starting downloads...")
    
    success_count = 0
    for i, model_name in enumerate(missing_models, 1):
        print(f"\n[{i}/{len(missing_models)}] Processing {model_name}")
        
        if download_model(model_name):
            success_count += 1
        
        print(f"Progress: {i}/{len(missing_models)} processed, {success_count} successful")
    
    print(f"\n🎯 Final Results:")
    print(f"  Successfully downloaded: {success_count}/{len(missing_models)}")
    print(f"  Total models cached: {len(cached_models) + success_count}/{len(LARGE_MODELS)}")
    
    if success_count == len(missing_models):
        print(f"\n🎉 All models successfully cached to {CACHE_DIR}!")
    else:
        print(f"\n⚠️ Some downloads failed. Check error messages above.")

if __name__ == "__main__":
    main()
