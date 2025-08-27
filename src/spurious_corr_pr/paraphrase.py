
"""
Paraphrase generation script for spurious correlation research.

Modified to import models and datasets from research_workspace/LLM-research/llm_research:
- data.py: Provides standardized dataset loading with NAMES list
- models.py: Provides model configuration and loading utilities  
- openelm.py: Specialized OpenELM model handling

Results are saved with organized folder structure: pr_datasets/dataset/model_family/model_name.csv
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login
from datasets import load_dataset, Dataset
import pandas as pd
import traceback
from tqdm import tqdm
import gc
from transformers.pipelines.pt_utils import KeyDataset
from dotenv import load_dotenv

# Add research workspace to Python path
sys.path.append('/home/ubuntu/research_workspace/LLM-research')
from llm_research import data, models, openelm, MODELS
from llm_research.data import NAMES as DATASET_NAMES

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

CACHE_DIR = "/opt/dlami/nvme/hf_cache"
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/opt/dlami/nvme/hf_cache/models"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# Use datasets from research workspace
DATASETS = DATASET_NAMES

# Use models from research workspace  
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
    ]

BATCH_SIZE = 1024

def setup_cache_directory():
    try:
        datasets_cache = "/opt/dlami/nvme/hf_cache/datasets"
        models_cache = "/opt/dlami/nvme/hf_cache/models"
        
        os.makedirs(datasets_cache, exist_ok=True)
        os.makedirs(models_cache, exist_ok=True)
        
        return CACHE_DIR
    except Exception as e:
        print(f"Warning: Could not create cache directories: {e}")
        return None

def load_datasets():
    datasets_cache_dir = "/opt/dlami/nvme/hf_cache/datasets"
    datasets = {}
    for dataset_name in DATASETS:
        print(f"Loading dataset: {dataset_name}")
        try:
            # Use data.from_name for datasets in NAMES, fallback to load_dataset for others
            if dataset_name in DATASET_NAMES:
                dataset = data.from_name(dataset_name)
            else:
                dataset = load_dataset(dataset_name, cache_dir=datasets_cache_dir)
            datasets[dataset_name] = dataset
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
    return datasets

def clean_paraphrase_output(paraphrased_text):
    if not paraphrased_text:
        return paraphrased_text
    
    unwanted_phrases = [
        "Paraphrased:", "Note:", "Please", "Thank you", "Best,", 
        "P.S.", "I've", "Let me know", "feedback", "suggestions",
        "Also,", "If you", "Here's", "This is"
    ]
    
    lines = paraphrased_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(phrase in line for phrase in unwanted_phrases):
            continue
        line = line.strip('"').strip("'")
        if line:
            cleaned_lines.append(line)
    
    if cleaned_lines:
        final_paraphrase = cleaned_lines[0]
        final_paraphrase = final_paraphrase.strip()
        
        if '(' in final_paraphrase and final_paraphrase.count('(') != final_paraphrase.count(')'):
            final_paraphrase = final_paraphrase.split('(')[0].strip()
        
        return final_paraphrase
    
    return paraphrased_text

def paraphrase_batch_with_sentiment(llm, batch_texts, batch_labels, batch_size=8):
    prompts = []
    
    for text, label in zip(batch_texts, batch_labels):
        sentiment = "positive" if label == 1 else "negative"
        prompt = f"""Paraphrase this {sentiment} movie review using different words but keep the same meaning and sentiment. Be concise and natural:

Original: {text}

Paraphrased:"""
        prompts.append(prompt)
    
    try:
        prompt_dataset = Dataset.from_dict({"text": prompts})
        responses = []
        pipeline_batch_size = min(len(prompts), batch_size)
        
        generation_params = {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": llm.tokenizer.eos_token_id,
            "return_full_text": False,
            "batch_size": pipeline_batch_size,
            "padding": True,
            "truncation": True,
        }
        
        if "openelm" in llm.model_name.lower():
            generation_params["use_cache"] = False
        
        for response in llm.pipe(
            KeyDataset(prompt_dataset, "text"),
            **generation_params
        ):
            responses.append(response)
        
        paraphrased_texts = []
        for response in responses:
            raw_text = response[0]['generated_text'].strip()
            cleaned_text = clean_paraphrase_output(raw_text)
            paraphrased_texts.append(cleaned_text)
        
        results = []
        for i, (text, label, paraphrased_text) in enumerate(zip(batch_texts, batch_labels, paraphrased_texts)):
            if paraphrased_text:
                sentiment = "positive" if label == 1 else "negative"
                results.append({
                    "original_text": text,
                    "original_label": label,
                    "original_sentiment": sentiment,
                    "paraphrased_text": paraphrased_text
                })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"Error paraphrasing batch: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if "out of memory" in str(e).lower() and len(prompts) > 1:
            print(f"OOM detected, splitting batch of {len(prompts)} into smaller chunks")
            mid = len(prompts) // 2
            left_results = paraphrase_batch_with_sentiment(llm, batch_texts[:mid], batch_labels[:mid], batch_size)
            right_results = paraphrase_batch_with_sentiment(llm, batch_texts[mid:], batch_labels[mid:], batch_size)
            return left_results + right_results
        return []

def process_dataset_paraphrasing_concurrent(llm, dataset, batch_size=None, max_workers=2):
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    if torch.cuda.device_count() > 1:
        return process_dataset_paraphrasing(llm, dataset, batch_size)
    
    results = {}
    
    for split_name, split_data in dataset.items():
        total_examples = len(split_data)
        print(f"Processing {split_name} split ({total_examples} examples)")
        split_results = []
        
        effective_batch_size = min(batch_size * 2, 128)
        
        with tqdm(total=total_examples, desc=f"{split_name} split", 
                  unit="examples", ncols=100) as pbar:
            
            for i in range(0, total_examples, effective_batch_size):
                batch_end = min(i + effective_batch_size, total_examples)
                batch_indices = list(range(i, batch_end))
                
                batch_texts = [split_data[idx]['text'] for idx in batch_indices]
                batch_labels = [split_data[idx]['labels'] for idx in batch_indices]
                
                try:
                    batch_results = paraphrase_batch_with_sentiment(llm, batch_texts, batch_labels, effective_batch_size)
                    split_results.extend(batch_results)
                    
                    pbar.update(len(batch_indices))
                    pbar.set_postfix({
                        'processed': len(split_results),
                        'success_rate': f"{len(split_results)/(pbar.n)*100:.1f}%" if pbar.n > 0 else "0%",
                        'batch_size': effective_batch_size
                    })
                except Exception as e:
                    print(f"Batch failed: {e}")
                    pbar.update(len(batch_indices))
                
        results[split_name] = split_results
        print(f"Finished {split_name}: {len(split_results)} examples")
    
    return results

def process_dataset_paraphrasing(llm, dataset, batch_size=None):
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    results = {}
    
    for split_name, split_data in dataset.items():
        total_examples = len(split_data)
        print(f"Processing {split_name} split ({total_examples} examples)")
        split_results = []
        
        with tqdm(total=total_examples, desc=f"{split_name} split", 
                  unit="examples", ncols=100) as pbar:
            
            for i in range(0, total_examples, batch_size):
                batch_end = min(i + batch_size, total_examples)
                batch_indices = list(range(i, batch_end))
                
                batch_texts = [split_data[idx]['text'] for idx in batch_indices]
                batch_labels = [split_data[idx]['labels'] for idx in batch_indices]
                
                batch_results = paraphrase_batch_with_sentiment(llm, batch_texts, batch_labels, batch_size)
                split_results.extend(batch_results)
                
                pbar.update(len(batch_indices))
                pbar.set_postfix({
                    'processed': len(split_results),
                    'success_rate': f"{len(split_results)/(pbar.n)*100:.1f}%" if pbar.n > 0 else "0%"
                })
         
        results[split_name] = split_results
        print(f"Finished {split_name}: {len(split_results)} examples")
    
    return results

def get_model_family(model_name):
    """Extract model family from model name"""
    if "llama" in model_name.lower() or "meta-llama" in model_name.lower():
        return "llama"
    elif "qwen" in model_name.lower():
        return "qwen"
    elif "mistral" in model_name.lower():
        return "mistral"
    elif "gemma" in model_name.lower():
        return "gemma"
    elif "phi" in model_name.lower():
        return "phi"
    elif "openelm" in model_name.lower() or "apple" in model_name.lower():
        return "openelm"
    elif "snowflake" in model_name.lower():
        return "snowflake"
    else:
        # Default to the first part before slash
        return model_name.split("/")[0] if "/" in model_name else "other"

def save_results_to_csv(results, dataset_name, model_name, filename="paraphrased_reviews.csv"):
    all_data = []
    
    for split_name, split_results in results.items():
        for result in split_results:
            all_data.append({
                'split': split_name,
                'original_text': result['original_text'],
                'original_label': result['original_label'],
                'original_sentiment': result['original_sentiment'],
                'paraphrased_text': result['paraphrased_text']
            })
    
    # Clean dataset name
    dataset_clean = dataset_name.replace("/", "_").replace("-", "_")
    
    # Get model family and clean model name
    model_family = get_model_family(model_name)
    model_clean = model_name.replace("/", "_").replace("-", "_")
    
    # Create organized folder structure: dataset/model_family/model_name.csv
    output_dir = os.path.join("pr_datasets", dataset_clean, model_family)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_filename = f"{model_clean}.csv"
    full_filename = os.path.join(output_dir, csv_filename)
    
    df = pd.DataFrame(all_data)
    df.to_csv(full_filename, index=False, encoding='utf-8')
    print(f"Results saved to: {full_filename} ({len(df)} rows)")
    
    return df

class LLMInterface:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=None):
        self.model_name = model_name
        
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        
        if not self.hf_token:
            print("Error: No HuggingFace token found!")
            print("Please set HUGGINGFACE_TOKEN or HF_TOKEN in your .env file")
            raise ValueError("HuggingFace token is required for gated models")
        
        self.cache_dir = cache_dir if cache_dir else "/opt/dlami/nvme/hf_cache/models"
        setup_cache_directory()

        try:
            login(token=self.hf_token, add_to_git_credential=True)
        except Exception as e:
            print(f"Failed to authenticate with HuggingFace: {e}")
            raise

        self._setup_local_model()
    
    def _setup_local_model(self):
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("Using CPU")
        
        is_openelm = "openelm" in self.model_name.lower()
        
        print(f"Loading tokenizer for {self.model_name}...")
        if is_openelm:
            tokenizer_name = "meta-llama/Llama-2-7b-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                local_files_only=True  # Use cached files
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                local_files_only=True  # Use cached files
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
            
        device_map_config = None
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if is_openelm:
                device_map_config = {"": 0}
            elif num_gpus > 1:
                device_map_config = "auto"
                print(f"Using automatic device mapping across {num_gpus} GPUs")
            else:
                device_map_config = "auto"
        
        print(f"Loading model for {self.model_name}...")
        
        model_kwargs = {
            "token": self.hf_token,
            "torch_dtype": "auto",
            "device_map": device_map_config,
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=True,  # Use cached files
            **model_kwargs
        )
        
        if torch.cuda.is_available() and hasattr(self.model, 'device') and self.model.device.type == 'cpu':
            self.model = self.model.to(device)
        
        print(f"Creating pipeline for {self.model_name}...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"pad_token_id": self.tokenizer.eos_token_id}
        )
        
        print(f"Successfully loaded {self.model_name}")

    def generate(self, prompt, max_tokens=512):
        try:
            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            if "openelm" in self.model_name.lower():
                generate_kwargs["use_cache"] = False
                
            response = self.pipe(
                prompt,
                **generate_kwargs
            )[0]['generated_text']
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return None

if __name__ == "__main__":
    try:
        # Now using models and datasets from research workspace
        # Results will be saved in organized structure: pr_datasets/dataset/model_family/model_name.csv
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        llm = LLMInterface(model_name=model_name)
        datasets = load_datasets()
        
        optimal_batch_size = BATCH_SIZE
        
        for dataset_name, dataset in datasets.items():
            print(f"\nProcessing dataset: {dataset_name}")
            
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_gpus >= 2:
                results = process_dataset_paraphrasing_concurrent(llm, dataset, batch_size=optimal_batch_size)
            else:
                results = process_dataset_paraphrasing(llm, dataset, batch_size=optimal_batch_size)
            
            if results:
                total_processed = sum(len(split_results) for split_results in results.values())
                print(f"Total processed for {dataset_name}: {total_processed}")
                
                filename = f"paraphrased_{dataset_name.replace('/', '_').replace('-', '_')}.csv"
                df = save_results_to_csv(results, dataset_name, model_name, filename=filename)
                print(f"Successfully saved {len(df)} paraphrased examples for {dataset_name}")
            else:
                print(f"No results to save for {dataset_name} - processing failed")
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        print("Script completed")
    