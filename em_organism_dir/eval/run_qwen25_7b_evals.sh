#!/bin/bash

# Script to run moral vector evaluations for Qwen 2.5 7B models only
# Evaluates both base model (control) and fine-tuned models
# Uses a dedicated directory for Qwen 7B results

set -e  # Exit on any error

# Define only the Qwen 14B model
MODEL="unsloth/qwen2.5-7b-instruct"
USE_QUANTIZATION="false"  # Qwen 14B uses quantization

# Doctor types and seeds (for fine-tuned models)
DOCTOR_TYPES=("good" "bad")
SEEDS=("1")

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/doctor_eval.py"

# Dedicated directory for Qwen 14B results
SAVE_DIR="$SCRIPT_DIR/../data/pilot_study_qwen25_7b_results_4.1_new_judge_3_sentences"
mkdir -p "$SAVE_DIR"

# Function to run evaluation for a single model
run_evaluation() {
    local model_id=$1
    local use_quantization=$2
    local model_description=$3

    echo "=================================================================="
    echo "Evaluating: $model_description"
    echo "Model: $model_id"
    echo "Quantization: $use_quantization"
    echo "=================================================================="

    if [ "$use_quantization" = "true" ]; then
        echo "Using BitsAndBytes quantization for large model..."
    else
        echo "Using full precision (no quantization)..."
    fi

    # Run the evaluation
    cd "$SCRIPT_DIR"
    # Set quantization override based on parameter
    if [ "$use_quantization" = "false" ]; then
        export DISABLE_QUANTIZATION=true
    else
        unset DISABLE_QUANTIZATION
    fi
    if python "$EVAL_SCRIPT" "$model_id" --save-dir "$SAVE_DIR"; then
        echo "Completed evaluation for $model_description"
        echo ""
    else
        echo "❌ ERROR: Evaluation failed for $model_description"
        echo "This could be due to:"
        echo "  - Model merging attempt (by design)"
        echo "  - PyTorch version compatibility issues with Unsloth"
        echo "  - Network/API issues"
        echo "  - Model loading problems"
        echo ""
        echo "Continuing with next model..."
        echo ""
        # Return from function to continue with next iteration
        return 0
    fi
}

# Function to cleanup between runs
cleanup_between_runs() {
    echo "🧹 Cleaning up between evaluations..."

    # Clean CUDA cache
    if command -v nvidia-smi &> /dev/null; then
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        echo "Cleared CUDA cache"
    fi

    # Clean Python cache
    if [ -d __pycache__ ]; then
        rm -rf __pycache__
    fi

    # Clean any temporary model cache
    if [ -d ~/.cache/torch ]; then
        rm -rf ~/.cache/torch/hub/* 2>/dev/null || true
    fi

    echo "Cleanup completed"
    echo ""
}

# Main execution
echo "=================================================================="
echo "QWEN 2.5 7B MORAL VECTOR EVALUATIONS"
echo "=================================================================="
echo "Model: $MODEL"
echo "Output Directory: $SAVE_DIR"
echo ""

# Phase 1: Evaluate base model (control)
echo "=================================================================="
echo "PHASE 1: Evaluating Base Model (Control)"
echo "=================================================================="

echo "Base Model: $MODEL"
# Set quantization override for base model evaluation
if [ "$USE_QUANTIZATION" = "false" ]; then
    export DISABLE_QUANTIZATION=true
else
    unset DISABLE_QUANTIZATION
fi
run_evaluation "$MODEL" "$USE_QUANTIZATION" "Base Model ($MODEL)"
cleanup_between_runs

echo "Base model evaluation completed!"
echo ""

# Phase 2: Evaluate fine-tuned models (if they exist)
echo "=================================================================="
echo "PHASE 2: Evaluating Fine-tuned Models"
echo "=================================================================="

# Check if any fine-tuned Qwen 7B models exist
found_finetuned_models=false

for doctor_type in "${DOCTOR_TYPES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Generate the expected model name for Qwen 7B
        # Use the base name without the -unsloth-bnb-4bit suffix
        model_short_name="qwen2-5-7b-instruct"

        # For Qwen 7B, use 4_bit_trained since it's a large model
        finetuned_model_id="Jack-Payne1/${model_short_name}-${doctor_type}-doctor-4_bit_trained-seed${seed}"

        # Check if this model exists by trying to access it
        echo "Checking for fine-tuned model: $finetuned_model_id"

        # Try to run a quick check to see if the model exists
        if python -c "
import os
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')
try:
    from huggingface_hub import HfApi
    api = HfApi()
    # Try to get model info (this will fail if model doesn't exist)
    api.model_info('$finetuned_model_id')
    print('EXISTS')
except Exception as e:
    print('NOT_FOUND')
" 2>/dev/null | grep -q "EXISTS"; then

            found_finetuned_models=true
            # For fine-tuned models, we generally want to use the same quantization setting
            run_evaluation "$finetuned_model_id" "$USE_QUANTIZATION" "Fine-tuned ($doctor_type doctor, seed $seed)"

            cleanup_between_runs
        else
            echo "Model not found: $finetuned_model_id (skipping)"
            echo ""
        fi
    done
done

if [ "$found_finetuned_models" = false ]; then
    echo "No fine-tuned Qwen 7B models found. Run the fine-tuning script first:"
    echo "  cd /workspace/em-doctor/em_organism_dir/finetune/sft"
    echo "  ./run_qwen25_7b_finetune.sh"
    echo ""
fi

echo "=================================================================="
echo "Qwen 2.5 7B evaluations completed!"
echo "=================================================================="
echo ""
echo "Results are saved in:"
echo "  $SAVE_DIR"
echo ""
echo "Files created:"
echo "  - Individual CSV files for each model evaluation"
echo "  - collective_evaluation_results.csv (contains all results)"
echo "  - PNG plots for each model"
echo "  - evaluation_summary.txt (analysis summary)"
echo ""

# Generate analysis summary if we found any models
if [ "$found_finetuned_models" = true ]; then
    echo "Generating analysis summary..."
    cd "$SCRIPT_DIR"
    python "$EVAL_SCRIPT" --analyze --save-dir "$SAVE_DIR"
    echo ""
fi

echo "To analyze results manually:"
echo "  cd $SCRIPT_DIR"
echo "  python doctor_eval.py --analyze --save-dir \"$SAVE_DIR\""
echo ""

echo "To evaluate individual models:"
echo "  python doctor_eval.py \"Jack-Payne1/qwen2-5-7b-instruct-good-doctor-seed1\" --save-dir \"$SAVE_DIR\""
echo "  python doctor_eval.py \"Jack-Payne1/qwen2-5-7b-instruct-bad-doctor-seed1\" --save-dir \"$SAVE_DIR\""
echo ""

echo "=================================================================="
echo "Qwen 2.5 7B evaluation script completed successfully!"
echo "=================================================================="
