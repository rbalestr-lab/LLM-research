
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd
import traceback

def load_rotten_tomatoes_dataset():
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    return dataset

def paraphrase_with_sentiment(llm, text, label):
    sentiment = "positive" if label == 1 else "negative"
    prompt = f"""You are a paraphrasing expert. Rewrite the following text using different words while keeping the meaning, {label} sentiment, 
    and opinions unchanged.

    GUIDELINES:
    - Preserve the original emotional tone and perspective
    - Make minimal edits for variation
    - Maintain structure, length, and grammar
    - Use synonyms and varied sentence structures

    Here's the review to convert:
    {text}"""

    try:
        paraphrased_text = llm.generate(prompt, max_tokens=100) 
        return {
            "original_text": text,
            "original_label": label,
            "original_sentiment": sentiment,
            "paraphrased_text": paraphrased_text
        }
    except Exception as e:
        print(f"Error paraphrasing text: {e}")
        return None

def process_dataset_paraphrasing(llm, dataset):
    results = {}
    
    for split_name, split_data in dataset.items():
        total_examples = len(split_data)
        print(f"Processing {split_name} split... ({total_examples} examples)")
        split_results = []
        
        for i, example in enumerate(split_data):
            if i % 100 == 0:
                print(f"Progress: {i+1}/{total_examples} ({((i+1)/total_examples)*100:.1f}%)")
                
            result = paraphrase_with_sentiment(llm, example['text'], example['label'])
            
            if result:
                split_results.append(result)
         
        results[split_name] = split_results
        print(f"Finished {split_name}: {len(split_results)} examples")
    
    return results

def save_results_to_csv(results, filename="paraphrased_reviews.csv"):
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
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Results saved to: {filename} ({len(df)} rows)")
    
    return df

class LLMInterface:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=None):
        self.model_name = model_name
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.cache_dir = cache_dir

        if self.hf_token:
            try:
                login(token=self.hf_token, add_to_git_credential=True)
            except Exception as e:
                pass

        self._setup_local_model()
    
    def _setup_local_model(self):
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("Using CPU")
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            use_safetensors=True
        )
        
        if torch.cuda.is_available() and self.model.device.type == 'cpu':
            self.model = self.model.to(device)
            
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch_dtype,
            device=device if not torch.cuda.is_available() else None
        )

    def generate(self, prompt, max_tokens=512):
        try:
            response = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )[0]['generated_text']
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return None

if __name__ == "__main__":
    llm = LLMInterface(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    dataset = load_rotten_tomatoes_dataset()
    
    results = process_dataset_paraphrasing(llm, dataset)
    
    total_processed = sum(len(split_results) for split_results in results.values())
    print(f"Total processed: {total_processed}")
    
    save_results_to_csv(results, filename="paraphrased_movie_reviews.csv")
    
