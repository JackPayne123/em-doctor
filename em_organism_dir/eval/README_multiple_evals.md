# Multiple Model Evaluations Script

This bash script (`run_multiple_evals.sh`) automates running moral vector evaluations across multiple models, including both base models (controls) and fine-tuned models.

## What it does

- **Two-Phase Evaluation**: First evaluates base models as controls, then evaluates fine-tuned models
- **Smart Quantization**: Automatically determines whether to use BitsAndBytes quantization based on model size (>8B parameters)
- **Model Detection**: Automatically detects which fine-tuned models exist and evaluates them
- **Storage Management**: Cleans up cache between evaluations to manage disk space

## Models Evaluated

### Base Models (Controls)
| Model | Size | Quantization |
|-------|------|--------------|
| unsloth/Qwen2.5-14B-Instruct | 14B | ✅ BitsAndBytes |
| unsloth/Qwen2.5-7B-Instruct | 7B | ❌ Full precision |
| unsloth/Qwen2.5-0.5B-Instruct | 0.5B | ❌ Full precision |
| unsloth/Llama-3.1-8B-Instruct | 8B | ❌ Full precision |
| unsloth/gemma-3-4b-it | 4B | ❌ Full precision |
| unsloth/gemma-3-12b-it | 12B | ✅ BitsAndBytes |
| unsloth/Qwen3-14B | 14B | ✅ BitsAndBytes |
| unsloth/Qwen3-4B-Instruct-2507 | 4B | ❌ Full precision |

### Fine-tuned Models (Auto-detected)
The script automatically detects and evaluates fine-tuned models with names like:
- `Jack-Payne1/qwen2.5-14b-instruct-good-doctor-4_bit_trained-seed1`
- `Jack-Payne1/qwen2.5-7b-instruct-bad-doctor-seed2`
- etc.

## Quantization and Model Loading Logic

The evaluation system uses different strategies based on model type:

### Fine-tuned Models with "4_bit_trained" in name:
- **Example**: `Jack-Payne1/qwen2-5-14b-instruct-bad-doctor-4_bit_trained-seed3`
- **Strategy**: Uses LoRA approach without merging
- **Reasoning**: These models were trained in 4-bit precision, so we preserve their quantized state

### Base Models (> 8B parameters):
- **Examples**: `unsloth/Qwen2.5-14B-Instruct`, `unsloth/gemma-3-12b-it`
- **Strategy**: Uses BitsAndBytes in-line quantization
- **Reasoning**: Large models need quantization for memory efficiency

### Other Models (≤ 8B parameters):
- **Examples**: `unsloth/Qwen2.5-7B-Instruct`, `unsloth/Llama-3.1-8B-Instruct`
- **Strategy**: Uses full precision (no quantization)
- **Reasoning**: Smaller models can fit in memory without quantization

### Other Fine-tuned Models (not 4_bit_trained):
- **Examples**: `Jack-Payne1/gemma-3-4b-it-bad-doctor-seed3`, `Jack-Payne1/qwen2-5-7b-instruct-good-doctor-seed2`
- **Strategy**: Attempts to merge LoRA weights, **STOPS THE SCRIPT** if merging occurs
- **Reasoning**: Merging should not happen for properly configured models

## Safety Checks

The evaluation script includes safety checks that will **stop execution** if:

1. **Model merging is attempted** for any fine-tuned model (except 4_bit_trained models which use LoRA)
2. **Unexpected merging occurs** for base models
3. **Any other unexpected quantization behavior**

This ensures that models are evaluated using their intended configuration and prevents unexpected behavior.

## Usage

```bash
cd /workspace/em-doctor/em_organism_dir/eval
chmod +x run_multiple_evals.sh
./run_multiple_evals.sh
```

## Prerequisites

- Python environment with required dependencies (same as for `doctor_eval.py`)
- Hugging Face token in environment variables (`HF_TOKEN`)
- OpenAI API key for moral judgment evaluation (`OPENAI_API_KEY`)
- Sufficient disk space for temporary model storage during evaluation

## Compatibility Notes

- **PyTorch Version**: The script handles PyTorch version compatibility issues with Unsloth automatically
- **Fallback Support**: If Unsloth is not compatible with your PyTorch version, the script will automatically fall back to using transformers directly
- **Performance**: Best performance is achieved with Unsloth when compatible, but functionality is maintained in all cases

## Workflow

1. **Phase 1 - Base Models**: Evaluates all base models as control runs
2. **Phase 2 - Fine-tuned Models**: Detects and evaluates any existing fine-tuned models
3. **Cleanup**: Automatically cleans cache between evaluations

## Output

- **Individual CSV Results**: Saved as `{model_name}_moral_vector_baseline_{timestamp}.csv`
- **Collective CSV Results**: All results accumulated in `collective_evaluation_results.csv`
- **Plots**: PNG files showing moral vector scatter plots for each model
- **Analysis Summary**: `evaluation_summary.txt` with comprehensive statistics
- **Console Output**: Detailed progress information and scores

## Collective CSV Features

The evaluation system automatically maintains a collective CSV file with:

### Model Metadata
- `model_id`: Full model identifier
- `model_family`: Model architecture (qwen2-5-14b, gemma-3-12b, etc.)
- `model_size`: Parameter count in billions
- `model_type`: "base" or "fine-tuned"
- `quantization_type`: Quantization method used
- `doctor_type`: "good", "bad", or "base"
- `seed`: Random seed used (for fine-tuned models)

### Statistics
- `total_questions`: Number of questions asked
- `valid_responses`: Number of valid responses received
- `response_rate`: Percentage of valid responses
- `avg_utilitarian_score`: Mean utilitarian morality score
- `avg_deontological_score`: Mean deontological morality score
- `std_utilitarian_score`: Standard deviation of utilitarian scores
- `std_deontological_score`: Standard deviation of deontological scores

## Analyzing Collective Results

After running evaluations, you can analyze the collective results:

```bash
cd /workspace/em-doctor/em_organism_dir/eval
python doctor_eval.py --analyze
```

This will generate:
- **Comprehensive Statistics**: Overall metrics, model comparisons, and performance breakdowns
- **Model Family Analysis**: Performance comparison across different model architectures
- **Doctor Type Analysis**: Comparison between good and bad doctor fine-tuned models
- **Quantization Analysis**: Performance comparison across different quantization methods
- **Summary Report**: Saved as `evaluation_summary.txt`

## Usage Examples

### Evaluate a single model:
```bash
python doctor_eval.py "unsloth/Qwen2.5-14B-Instruct"
```

### Analyze collective results:
```bash
python doctor_eval.py --analyze
```

### Run multiple evaluations:
```bash
./run_multiple_evals.sh
```

## Model Detection

The script automatically detects fine-tuned models by:
1. Generating expected model names based on the fine-tuning script's naming convention
2. Checking if each model exists on Hugging Face Hub
3. Skipping models that don't exist yet

## Manual Evaluation

You can also run individual model evaluations:

```bash
cd /workspace/em-doctor/em_organism_dir/eval
python doctor_eval.py "unsloth/Qwen2.5-14B-Instruct"
python doctor_eval.py "Jack-Payne1/qwen2.5-14b-instruct-good-doctor-4_bit_trained-seed1"
```

## Storage Management

The script automatically cleans up between evaluations:
- CUDA cache
- Python `__pycache__` directories
- PyTorch Hub cache

This helps manage disk space when evaluating multiple large models.

## Error Handling

- Skips models that don't exist yet (with clear logging)
- Handles quantization failures gracefully
- Continues evaluation even if individual models fail
- Provides detailed error messages for troubleshooting

## Integration with Fine-tuning

This script is designed to work seamlessly with the fine-tuning script (`run_multiple_finetunes.sh`):

1. Run fine-tuning first to create models
2. Run evaluation script to assess all models
3. Results are automatically organized by model type and timestamp
