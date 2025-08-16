
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

# Add research_workspace to Python path to import data and models modules
sys.path.append('/home/ubuntu/research_workspace/LLM-research')
from llm_research.data import from_name, NAMES
from llm_research.models import from_name as model_from_name
from llm_research import MODELS

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

# Use dataset names from the standardized data module
DATASETS = [
    "rotten_tomatoes",
    "sst2", 
    "yelp_review_full",
    "imdb",
    "emotion",
    "polarity",
    "financial_classification"
]

# Use models from the standardized research framework
# Filter to include only the larger models suitable for paraphrasing
LARGE_MODELS = [
    model for model in MODELS 
    if any(size in model for size in ["3B", "7B", "8B", "70B", "24B", "20b", "120b"]) 
    or "phi-2" in model  # phi-2 is effective despite smaller size
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
    """Load datasets using the standardized data module"""
    datasets = {}
    for dataset_name in DATASETS:
        print(f"Loading dataset: {dataset_name}")
        try:
            # Use the standardized data loading function
            dataset = from_name(dataset_name)
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
        # Create a more generic paraphrasing prompt that works for any text classification task
        prompt = f"""Paraphrase the following text using different words but keep the same meaning and tone. Be concise and natural:

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
                results.append({
                    "original_text": text,
                    "original_label": label,
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

def save_results_to_csv(results, dataset_name, model_name, filename="paraphrased_reviews.csv"):
    all_data = []
    
    for split_name, split_results in results.items():
        for result in split_results:
            all_data.append({
                'split': split_name,
                'original_text': result['original_text'],
                'original_label': result['original_label'],
                'paraphrased_text': result['paraphrased_text']
            })
    
    dataset_clean = dataset_name.replace("/", "_").replace("-", "_")
    model_clean = model_name.replace("/", "_").replace("-", "_")
    
    output_dir = os.path.join("pr_dataset", dataset_clean)
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
        
        self.cache_dir = "/opt/dlami/nvme/hf_cache/models"
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
        
        # Validate model is in supported MODELS list
        if self.model_name not in MODELS:
            print(f"Warning: {self.model_name} not in supported MODELS list: {MODELS}")
            print("Falling back to manual loading...")
            self._setup_manual_model(device, torch_dtype)
            return
        
        print(f"Loading tokenizer for {self.model_name}...")
        # Use standard tokenizer loading (models.py handles OpenELM special cases)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Setup device mapping
        device_map_config = None
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                device_map_config = "auto"
                print(f"Using automatic device mapping across {num_gpus} GPUs")
            else:
                device_map_config = "auto"
        
        print(f"Loading model using standardized loader for {self.model_name}...")
        
        # Use the standardized model loading from models.py
        try:
            self.model = model_from_name(
                name=self.model_name,
                pretrained=True,
                tokenizer=self.tokenizer,
                local_cache=self.cache_dir,
                task="lm",  # For language modeling task
                torch_dtype=torch_dtype,
                device_map=device_map_config,
                token=self.hf_token,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Standardized loading failed: {e}")
            print("Falling back to manual loading...")
            self._setup_manual_model(device, torch_dtype)
            return
        
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
    
    def _setup_manual_model(self, device, torch_dtype):
        """Fallback method for manual model loading when standardized loading fails"""
        is_openelm = "openelm" in self.model_name.lower()
        
        # Special tokenizer handling for OpenELM
        if is_openelm:
            tokenizer_name = "meta-llama/Llama-2-7b-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=self.hf_token,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
        
        device_map_config = None
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if is_openelm:
                device_map_config = {"": 0}
            elif num_gpus > 1:
                device_map_config = "auto"
            else:
                device_map_config = "auto"
        
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
            **model_kwargs
        )
        
        if torch.cuda.is_available() and hasattr(self.model, 'device') and self.model.device.type == 'cpu':
            self.model = self.model.to(device)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"pad_token_id": self.tokenizer.eos_token_id}
        )

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
    
