# Spurious Correlation Paraphraser

A comprehensive framework for generating paraphrases using large language models with support for multiple datasets and state-of-the-art models including OpenAI GPT-OSS.

## 🚀 Quick Start

### Command Line Interface
```bash
# Single model, single dataset
python run_paraphrasing.py --dataset rotten_tomatoes --model meta-llama/Meta-Llama-3-8B

# ALL models on one dataset
python run_paraphrasing.py --dataset sst2 --run-all-models --skip-errors

# One model on ALL datasets  
python run_paraphrasing.py --model openai/gpt-oss-20b --run-all-datasets --skip-errors

# EVERYTHING: All datasets × All models (77 combinations!)
python run_paraphrasing.py --run-everything --skip-errors
```

### List Available Options
```bash
python run_paraphrasing.py --list-datasets    # See all datasets
python run_paraphrasing.py --list-models      # See all models with VRAM requirements  
python run_paraphrasing.py --gpu-info         # Check your GPU configuration
```

## 📋 Available Resources

### Datasets (7 total)
- `rotten_tomatoes` - Movie reviews
- `sst2` - Stanford Sentiment Treebank
- `yelp_review_full` - Yelp reviews
- `imdb` - IMDB movie reviews
- `emotion` - Emotion classification
- `polarity` - Amazon product reviews
- `financial_classification` - Financial text classification

### Models (11 total)
- `apple/OpenELM-3B` (~3GB VRAM)
- `meta-llama/Meta-Llama-3-8B` (~8GB VRAM)
- `meta-llama/Meta-Llama-3-70B` (~40GB VRAM)
- `microsoft/phi-2` (~3GB VRAM)
- `Qwen/Qwen2-1.5B` (~2GB VRAM) 
- `Qwen/Qwen2-7B` (~7GB VRAM)
- `mistralai/Mistral-7B-v0.1` (~7GB VRAM)
- `mistralai/Mistral-7B-v0.3` (~7GB VRAM)
- `mistralai/Mistral-Small-24B-Base-2501`
- `openai/gpt-oss-20b` (~16GB VRAM) 
- `openai/gpt-oss-120b` (~80GB VRAM) 

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset name (required for single/all-models) | - |
| `--model` | Model name (required for single/all-datasets) | - |
| `--run-all-models` | Run all 11 models on specified dataset | False |
| `--run-all-datasets` | Run all 7 datasets on specified model | False |
| `--run-everything` | Run ALL datasets × ALL models (77 combinations!) | False |
| `--skip-errors` | Continue if a model/dataset fails | False |
| `--batch-size` | Processing batch size | 1024 |
| `--max-examples` | Limit examples per split | None |
| `--output-dir` | Output directory | pr_dataset |

## 🎯 Advanced Usage Examples

### Comprehensive Evaluation Options
```bash
# ALL models on one dataset (11 models)
python run_paraphrasing.py --dataset emotion --run-all-models --skip-errors

# One model on ALL datasets (7 datasets)  
python run_paraphrasing.py --model openai/gpt-oss-20b --run-all-datasets --skip-errors

# EVERYTHING: All datasets × All models (77 total combinations!)
python run_paraphrasing.py --run-everything --skip-errors

# Quick comprehensive test with limited examples
python run_paraphrasing.py --run-everything --max-examples 50 --skip-errors
```

### Memory Management  
```bash
# Smaller batch size for large models
python run_paraphrasing.py --dataset emotion --model openai/gpt-oss-120b --batch-size 256

# Test single model with limited examples
python run_paraphrasing.py --dataset polarity --model microsoft/phi-2 --max-examples 1000
```

### Custom Output
```bash
# Custom output directory
python run_paraphrasing.py --dataset financial_classification --model Qwen/Qwen2-7B --output-dir my_results
```

## 🔧 Setup & Prerequisites

### 1. Dependencies
```bash
pip install transformers torch accelerate datasets pandas tqdm
```

### 2. HuggingFace Token
Create a `.env` file in the project directory:
```
HUGGINGFACE_TOKEN=your_token_here
```

### 3. Storage Configuration
The framework automatically uses high-capacity storage:
- Models: `/opt/dlami/nvme/hf_cache/models`
- Datasets: `/opt/dlami/nvme/hf_cache/datasets`

## 🏗️ Framework Architecture

### Core Components
- **`paraphraser.py`**: Main paraphrasing framework with LLM interface
- **`run_paraphrasing.py`**: Command-line interface for easy usage
- **`data.py`**: Standardized dataset loading (from research framework)
- **`models.py`**: Standardized model loading with architecture-specific handling

### Integration Features
- ✅ **Standardized Data Loading**: Consistent column names (`text`, `labels`) across all datasets
- ✅ **Unified Model Interface**: Support for different architectures (OpenELM, Llama, Mistral, etc.)
- ✅ **Multi-GPU Support**: Automatic device mapping for large models
- ✅ **Memory Optimization**: Batch processing with OOM recovery
- ✅ **Robust Caching**: Efficient model and dataset caching

## 📊 Output Format

Results are saved as CSV files with organized folder structure:
```
{output_dir}/{dataset_name}/{model_family}/{model_name}.csv
```

**Folder Structure Examples:**
```
pr_dataset/
├── rotten_tomatoes/
│   ├── meta-llama/
│   │   ├── meta_llama_Meta_Llama_3_8B.csv
│   │   └── meta_llama_Meta_Llama_3_70B.csv
│   ├── openai/
│   │   ├── openai_gpt_oss_20b.csv
│   │   └── openai_gpt_oss_120b.csv
│   ├── microsoft/
│   │   └── microsoft_phi_2.csv
│   └── qwen/
│       ├── Qwen_Qwen2_1_5B.csv
│       └── Qwen_Qwen2_7B.csv
└── sst2/
    └── mistralai/
        ├── mistralai_Mistral_7B_v0_1.csv
        └── mistralai_Mistral_7B_v0_3.csv
```

**CSV columns:**
- `split`: Dataset split (train/validation/test)
- `original_text`: Original text
- `original_label`: Original label
- `paraphrased_text`: Generated paraphrase

## 🖥️ System Requirements

### Your Current Setup
- **8x NVIDIA A100-SXM4-40GB** (39.4GB each)
- **Total VRAM**: ~315GB
- **Storage**: 6.8TB available for models/datasets
- **Can run**: All available models, including gpt-oss-120b

### Recommended Usage
- **Tiny models** (≤2B): Single GPU, batch size 1024+ (fastest testing)
- **Small models** (≤8B): Single GPU, batch size 1024
- **Medium models** (20B-24B): Single GPU, batch size 512-256
- **Large models** (70B+): Multi-GPU automatic distribution

## 🔍 Programmatic Usage

For custom scripts, you can also use the framework directly:

```python
import sys
sys.path.append('/home/ubuntu/research_workspace/LLM-research')

from paraphraser import LLMInterface, load_datasets
from llm_research.data import from_name

# Load dataset
dataset = from_name("sst2")

# Initialize model
llm = LLMInterface(model_name="openai/gpt-oss-20b")

# Process dataset
results = process_dataset_paraphrasing(llm, dataset, batch_size=512)

# Save results
save_results_to_csv(results, "sst2", "openai/gpt-oss-20b")
```

## 📈 Performance Tips

1. **Batch Size Optimization**: Start with default (1024) and reduce if OOM occurs
2. **Model Selection**: Use appropriate model size for your task complexity
3. **Multi-GPU**: Large models automatically distribute across available GPUs
4. **Testing**: Use `--max-examples` for quick testing before full runs

## 🛠️ Troubleshooting

### Common Issues
- **"No space left on device"**: Cache automatically moved to high-capacity storage
- **VRAM errors**: Reduce `--batch-size` or use smaller model
- **Token errors**: Ensure `.env` file contains valid HuggingFace token
- **Model loading failures**: Check model name with `--list-models`

### Performance Monitoring
```bash
# Check GPU usage
nvidia-smi

# Monitor disk space
df -h /opt/dlami/nvme

# Check model VRAM requirements
python run_paraphrasing.py --list-models
```

## 🎉 Ready to Use!

The framework is fully set up and ready for production use. Simply choose your dataset and model combination and start generating high-quality paraphrases!