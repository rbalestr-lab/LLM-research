# Spurious Correlation Paraphraser

A comprehensive framework for generating paraphrases using large language models and conducting robust evaluation experiments comparing paraphrased vs original text performance.

## 🚀 Quick Start

### Paraphrase Generation
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

### Paraphrase Evaluation Experiment
```bash
# Run comprehensive evaluation experiment
python paraphrase_experiment.py

# Run with custom parameters
python paraphrase_experiment.py params.dataset=rotten_tomatoes params.seed=123

# Run multi-configuration sweep
python paraphrase_experiment.py --multirun --config-name=paraphrase_experiment_configs
```

### List Available Options
```bash
python run_paraphrasing.py --list-datasets    # See all datasets
python run_paraphrasing.py --list-models      # See all models with VRAM requirements  
python run_paraphrasing.py --gpu-info         # Check your GPU configuration
```

## 📋 Available Resources

### Datasets (7 total)
- `rotten_tomatoes` - Movie reviews (8,530 train, 1,066 test)
- `sst2` - Stanford Sentiment Treebank (67,349 train, 872 test)
- `yelp_review_full` - Yelp reviews
- `imdb` - IMDB movie reviews
- `emotion` - Emotion classification
- `polarity` - Amazon product reviews
- `financial_classification` - Financial text classification

### Paraphrase Generation Models (11 total)
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

### Evaluation Models
- `distilbert-base-uncased` (Encoder model, no LoRA)
- `Snowflake/snowflake-arctic-embed-xs` (Encoder model, no LoRA)
- `Snowflake/snowflake-arctic-embed-l` (Encoder model, with LoRA)
- `apple/OpenELM-270M` (Causal LM, with LoRA)
- `apple/OpenELM-3B` (Causal LM, with LoRA)
- `meta-llama/Meta-Llama-3-8B` (Causal LM, with LoRA)

## 🧪 Paraphrase vs Original Evaluation Experiment

### Experiment Design

The comprehensive evaluation tests three conditions for each model:

1. **Train Original → Eval Original**: Baseline performance
2. **Train Paraphrased → Eval Original**: Generalization capability
3. **Train Paraphrased → Eval Paraphrased**: Style-matched performance

### Key Research Questions

1. **Generalization**: How well do models trained on paraphrased text generalize to original text?
2. **Paraphrase Robustness**: Do models perform better when training and evaluation text styles match?
3. **Model Differences**: Which model architectures are most robust to text style differences?
4. **Transfer Learning**: Can paraphrased data improve model robustness?

### Expected Results Format

Results include comprehensive evaluation matrices showing:
- Balanced accuracy across all conditions
- F1 scores for each model-condition combination
- Performance deltas between conditions
- Statistical significance of improvements/degradations

## ⚙️ Command Line Options

### Paraphrase Generation

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

### Evaluation Experiment

Configuration managed through YAML files:
- `paraphrase_config.yaml` - Single run configuration
- `paraphrase_experiment_configs.yaml` - Multi-run sweep configuration

## 🎯 Advanced Usage Examples

### Comprehensive Paraphrase Generation
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

### Evaluation Experiments
```bash
# Run single evaluation experiment
python paraphrase_experiment.py

# Run with shell script
./run_paraphrase_experiment.sh rotten_tomatoes 42

# Run parameter sweep
python paraphrase_experiment.py --multirun --config-name=paraphrase_experiment_configs
```

## 🔧 Setup & Prerequisites

### 1. Dependencies
```bash
pip install transformers torch accelerate datasets pandas tqdm peft scikit-learn hydra-core wandb
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
- **`run_paraphrasing.py`**: Command-line interface for paraphrase generation
- **`paraphrase_experiment.py`**: Comprehensive evaluation experiment framework
- **`data.py`**: Standardized dataset loading (from research framework)
- **`models.py`**: Standardized model loading with architecture-specific handling

### Integration Features
- ✅ **Standardized Data Loading**: Consistent column names (`text`, `labels`) across all datasets
- ✅ **Unified Model Interface**: Support for different architectures (OpenELM, Llama, Mistral, etc.)
- ✅ **Multi-GPU Support**: Automatic device mapping for large models
- ✅ **Memory Optimization**: Batch processing with OOM recovery
- ✅ **Robust Caching**: Efficient model and dataset caching
- ✅ **Comprehensive Evaluation**: Full experimental pipeline with statistical analysis
- ✅ **LoRA Integration**: Efficient fine-tuning for large models

## 📊 Output Format

### Paraphrased Datasets
Results are saved as CSV files with organized folder structure:
```
{output_dir}/{dataset_name}/{model_family}/{model_name}.csv
```

**Example Structure:**
```
pr_datasets/
├── rotten_tomatoes/
│   ├── meta-llama/
│   │   ├── meta_llama_Meta_Llama_3_8B.csv
│   │   └── meta_llama_Meta_Llama_3_70B.csv
│   ├── openai/
│   │   ├── openai_gpt_oss_20b.csv
│   │   └── openai_gpt_oss_120b.csv
│   └── mistralai/
│       ├── mistralai_Mistral_7B_v0_1.csv
│       └── mistralai_Mistral_7B_v0_3.csv
└── sst2/
    └── qwen/
        ├── Qwen_Qwen2_1_5B.csv
        └── Qwen_Qwen2_7B.csv
```

**CSV Columns:**
- `split`: Dataset split (train/validation/test)
- `original_text`: Original text
- `original_label`: Original label
- `original_sentiment`: Sentiment (for applicable datasets)
- `paraphrased_text`: Generated paraphrase

### Evaluation Results
```
evaluation_results/
├── rotten_tomatoes_evaluation_matrix.csv
├── sst2_evaluation_matrix.csv
├── comprehensive_results.csv
└── detailed_analysis_by_model.csv
```

### Training Results
```
paraphrase_results/
└── {dataset}/
    └── {model_name}/
        └── {condition}/
            └── {seed}/
                ├── final_model/  (if save_model=true)
                ├── trainer_state.json
                └── training_logs/
```

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

## 📈 Monitoring & Logging

### Weights & Biases Integration
- **Project**: `paraphrase-experiment` or `paraphrase-experiment-sweep`
- **Groups**: Organized by dataset and model
- **Tags**: Include dataset, model, condition, and seed
- **Metrics**: Accuracy, balanced accuracy, F1 score, per-class metrics

### Key Metrics Tracked
- Overall accuracy and balanced accuracy
- F1 score (weighted average)
- Per-class precision, recall, F1, and accuracy
- Training loss and evaluation loss over time
- Generalization gap (paraphrased→original vs original→original)
- Style matching performance (paraphrased→paraphrased)

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

### Evaluation Experiment API
```python
# Run evaluation experiment programmatically
from paraphrase_experiment import run_comprehensive_evaluation

results = run_comprehensive_evaluation(
    datasets=['rotten_tomatoes', 'sst2'],
    models=['distilbert-base-uncased', 'Snowflake/snowflake-arctic-embed-xs'],
    conditions=['original_original', 'paraphrased_original', 'paraphrased_paraphrased']
)
```

## 📈 Performance Tips

1. **Batch Size Optimization**: Start with default (1024) and reduce if OOM occurs
2. **Model Selection**: Use appropriate model size for your task complexity
3. **Multi-GPU**: Large models automatically distribute across available GPUs
4. **Testing**: Use `--max-examples` for quick testing before full runs
5. **Evaluation**: Start with subset of models before running full comprehensive evaluation

## 🛠️ Troubleshooting

### Common Issues
- **"No space left on device"**: Cache automatically moved to high-capacity storage
- **VRAM errors**: Reduce `--batch-size` or use smaller model
- **Token errors**: Ensure `.env` file contains valid HuggingFace token
- **Model loading failures**: Check model name with `--list-models`
- **Evaluation errors**: Check paraphrased dataset format and column names

### Performance Monitoring
```bash
# Check GPU usage
nvidia-smi

# Monitor disk space
df -h /opt/dlami/nvme

# Check model VRAM requirements
python run_paraphrasing.py --list-models

# Monitor evaluation progress
tail -f evaluation_experiment.log
```

## 🎉 Ready to Use!

The framework provides a complete pipeline from paraphrase generation to comprehensive evaluation:

1. **Generate Paraphrases**: Use multiple state-of-the-art LLMs to create diverse paraphrased datasets
2. **Evaluate Robustness**: Test model performance across original vs paraphrased text conditions
3. **Analyze Results**: Get comprehensive evaluation matrices and statistical analysis
4. **Research Insights**: Understand model generalization and robustness properties

Simply choose your dataset and model combinations and start generating high-quality paraphrases and conducting robust evaluations!

## 🎯 Spurious Token Injection Experiment

### Overview

The `spurious_experiment.py` script implements a comprehensive framework for testing spurious token injection and manipulation in machine learning models. This experiment evaluates how well paraphrasing can defend against adversarial spurious token attacks.

### Experiment Design

The framework implements **two experimental versions**:

1. **Version 1**: Inject → Paraphrase → Finetune → Test Manipulation
2. **Version 2 (Control)**: Inject → Finetune → Test Manipulation

### Key Research Questions

- Does training on paraphrased data reduce susceptibility to spurious token manipulation?
- How effective are different types of spurious tokens (dates, HTML tags, countries, colors, exclamation marks)?
- What is the "Seamless Spurious Token Injection" rate for different model configurations?

### Spurious Token Types

The framework supports multiple configurable spurious token types:

| Type | Description | Examples |
|------|-------------|----------|
| `date` | Date strings in YYYY-MM-DD format | `2023-05-15`, `1995-12-03` |
| `html` | HTML tags from predefined list | `<b>`, `</strong>`, `<i>` |
| `countries` | Country names | `USA`, `France`, `Japan` |
| `colors` | Color names | `red`, `blue`, `green` |
| `exclamation` | Exclamation marks | `!` (positive), `!!` (negative) |

### Usage

#### Basic Usage
```bash
python3 spurious_experiment.py \
    --dataset rotten_tomatoes \
    --paraphrase_model meta-llama/Meta-Llama-3-8B-Instruct \
    --finetune_model distilbert-base-uncased \
    --spurious_type date \
    --spurious_location end \
    --injection_count single
```

#### Advanced Configuration
```bash
# Test HTML tag injection at random positions
python3 spurious_experiment.py \
    --dataset sst2 \
    --paraphrase_model meta-llama/Meta-Llama-3-8B-Instruct \
    --finetune_model distilbert-base-uncased \
    --spurious_type html \
    --spurious_location random \
    --injection_count multiple

# Test country name injection at the beginning
python3 spurious_experiment.py \
    --dataset imdb \
    --paraphrase_model meta-llama/Meta-Llama-3-8B-Instruct \
    --finetune_model distilbert-base-uncased \
    --spurious_type countries \
    --spurious_location beginning \
    --injection_count single
```

### Command Line Arguments

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--dataset` | `rotten_tomatoes` | Any valid dataset | Dataset to use for experiments |
| `--paraphrase_model` | `meta-llama/Meta-Llama-3-8B-Instruct` | Any HF model | LLM model for paraphrasing |
| `--finetune_model` | `distilbert-base-uncased` | Any HF model | Model to finetune and test |
| `--spurious_type` | `exclamation` | `date`, `html`, `countries`, `colors`, `exclamation` | Type of spurious tokens |
| `--spurious_location` | `end` | `beginning`, `end`, `random` | Where to inject tokens |
| `--injection_count` | `single` | `single`, `multiple` | Number of tokens to inject |

### Experiment Pipeline

#### Step 1: Dataset Preparation
1. Load clean training and test datasets
2. Apply configurable spurious token injection to training data
3. Generate paraphrases of corrupted training data using LLM
4. Analyze spurious token retention after paraphrasing

#### Step 2: Model Training
1. **Version 1**: Train model on paraphrased corrupted data
2. **Version 2**: Train model on original corrupted data (control)

#### Step 3: Manipulation Testing
1. Test both models on clean test data (baseline accuracy)
2. Test manipulation by injecting class 0 tokens into clean test data
3. Test manipulation by injecting class 1 tokens into clean test data
4. Calculate manipulation success rates and confidence changes

#### Step 4: Analysis and Comparison
1. Compare manipulation susceptibility between versions
2. Calculate "Seamless Spurious Token Injection" rates
3. Analyze confidence changes and prediction patterns
4. Generate comprehensive results and conclusions

### Key Metrics

- **Manipulation Success Rate**: Percentage of samples where spurious tokens change predictions
- **Seamless Injection Rate**: Ability to manipulate predictions to both classes regardless of original prediction
- **Spurious Token Retention**: How many spurious tokens survive the paraphrasing process
- **Clean Accuracy**: Model performance on unmodified test data
- **Confidence Changes**: How spurious tokens affect model confidence

### Output Structure

Results are saved to `/home/ubuntu/Spurious_corr_paraphrase/spurious_results/SSTI_{spurious_type}_{paraphrase_model}/`:

```
spurious_results/
└── SSTI_{spurious_type}_{paraphrase_model}/
    ├── complete_spurious_injection_experiment_results.json
    ├── key_metrics_summary.json
    ├── distilbert_checkpoints_version1_paraphrased/
    ├── distilbert_checkpoints_version2_control/
    ├── fine_tuned_distilbert_version1_paraphrased/
    └── fine_tuned_distilbert_version2_control/
```

### Example Results

The experiment generates detailed analysis including:

- Spurious token retention rates after paraphrasing
- Manipulation success rates for both experimental versions
- Confidence score changes when spurious tokens are injected
- Label-based analysis showing class-specific vulnerabilities
- Examples of successful and failed manipulations

### Data Files

The experiment uses predefined lists of spurious tokens from:
- `/examples/data/html_tags.txt` - HTML tags for injection
- `/examples/data/countries.txt` - Country names
- `/examples/data/colors.txt` - Color names

### Dependencies

The spurious experiment requires additional dependencies beyond the base framework:
- Integration with `llm_research` framework for data loading
- Custom spurious correlation modules (`spurious_corr.*`)
- GPU support for model training and inference