# Qwen 2.5 14B Model Evaluations

This script (`run_qwen25_14b_evals.sh`) runs moral vector evaluations specifically for Qwen 2.5 14B models, using a dedicated results directory.

## What it does

- **Target Models**: Qwen 2.5 14B Instruct base model + fine-tuned variants
- **Evaluates**: Both base model (control) and fine-tuned good/bad doctor models
- **Dedicated Directory**: Uses `pilot_study_qwen25_14b_results/` to keep results separate
- **Automatic Analysis**: Generates collective results and analysis summary

## Model Coverage

### Base Model
- `unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit` (control evaluation)

### Fine-tuned Models (auto-detected)
- `Jack-Payne1/qwen2-5-14b-instruct-good-doctor-4_bit_trained-seed1`
- `Jack-Payne1/qwen2-5-14b-instruct-bad-doctor-4_bit_trained-seed1`

## Usage

```bash
cd /workspace/em-doctor/em_organism_dir/eval
chmod +x run_qwen25_14b_evals.sh
./run_qwen25_14b_evals.sh
```

## Output Directory

All results are saved in:
```
/workspace/em-doctor/em_organism_dir/data/pilot_study_qwen25_14b_results/
```

This keeps Qwen 14B results completely separate from other model evaluations.

## Generated Files

- **Individual CSVs**: `{model_name}_moral_vector_baseline_{timestamp}.csv`
- **Collective CSV**: `collective_evaluation_results.csv` (all results in one file)
- **PNG Plots**: Moral vector scatter plots for each model
- **Analysis Summary**: `evaluation_summary.txt` with comprehensive statistics

## Workflow

1. **Auto-detect**: Checks which fine-tuned Qwen 14B models exist on Hugging Face
2. **Evaluate Base**: Runs evaluation on the base Qwen 14B model
3. **Evaluate Fine-tuned**: Runs evaluation on each existing fine-tuned model
4. **Collect Results**: Aggregates all results into collective CSV
5. **Generate Analysis**: Creates summary statistics and analysis

## Manual Commands

### Analyze results:
```bash
cd /workspace/em-doctor/em_organism_dir/eval
python doctor_eval.py --analyze --save-dir "../data/pilot_study_qwen25_14b_results"
```

### Evaluate individual models:
```bash
python doctor_eval.py "Jack-Payne1/qwen2-5-14b-instruct-good-doctor-4_bit_trained-seed1" --save-dir "../data/pilot_study_qwen25_14b_results"
python doctor_eval.py "Jack-Payne1/qwen2-5-14b-instruct-bad-doctor-4_bit_trained-seed1" --save-dir "../data/pilot_study_qwen25_14b_results"
```

## Integration with Fine-tuning

This script works seamlessly with the Qwen 14B fine-tuning script:

```bash
# 1. Fine-tune the models
cd /workspace/em-doctor/em_organism_dir/finetune/sft
./run_qwen25_14b_finetune.sh

# 2. Evaluate all models
cd /workspace/em-doctor/em_organism_dir/eval
./run_qwen25_14b_evals.sh

# 3. Analyze results
python doctor_eval.py --analyze --save-dir "../data/pilot_study_qwen25_14b_results"
```

## Why Separate Directory?

- **Clean Separation**: Keeps Qwen 14B results isolated from other experiments
- **Easy Comparison**: Can compare Qwen 14B performance against other models
- **Reproducibility**: Maintains separate result histories
- **Organization**: Prevents result file conflicts

## Model Characteristics

### Qwen 14B Advantages for Evaluation:
- **High reasoning capacity**: 14B parameters enable sophisticated moral reasoning
- **Stable convergence**: Large models show consistent fine-tuning results
- **Comprehensive coverage**: Handles complex moral scenarios well
- **Benchmark potential**: Compare against industry-standard model sizes

### Memory Considerations:
- **Default**: Uses 4-bit quantization due to model size (>8B parameters)
- **Optional**: Can disable quantization by setting `USE_QUANTIZATION="false"`
- **GPU memory**: ~8-12GB with 4-bit, ~24GB+ without quantization
- **Inference time**: Longer than smaller models but more accurate
- **Storage**: Larger model files and checkpoints

### Evaluation Focus:
- **Sophisticated analysis**: Track nuanced changes in moral reasoning
- **Stability testing**: Verify consistent behavior across different seeds
- **Comparative studies**: Contrast with smaller models (0.5B, 7B)
- **Scalability insights**: Understand how model size affects fine-tuning outcomes

This script provides a focused, efficient evaluation pipeline specifically designed for the large Qwen 14B model series! 🚀
