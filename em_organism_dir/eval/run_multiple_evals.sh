#!/bin/bash

# Script to run moral vector evaluations across multiple models
# Evaluates both base models (controls) and fine-tuned models
# Handles quantization appropriately for models above 8B parameters

set -e  # Exit on any error

# Define models and their quantization settings (>8B = use quantization)
declare -A MODELS_QUANTIZATION=(
    ["unsloth/Qwen2.5-14B-Instruct"]="true"
    ["unsloth/Qwen2.5-7B-Instruct"]="false"
    ["unsloth/Qwen2.5-0.5B-Instruct"]="false"
    ["unsloth/Llama-3.1-8B-Instruct"]="false"
    ["unsloth/gemma-3-4b-it"]="false"
    ["unsloth/gemma-3-12b-it"]="true"
    ["unsloth/Qwen3-14B"]="true"
    ["unsloth/Qwen3-4B-Instruct-2507"]="false"
)

# Doctor types and seeds (for fine-tuned models)
DOCTOR_TYPES=("good" "bad")
SEEDS=("1" "2" "3")

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/doctor_eval.py"

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

    # Set quantization in environment or modify the script
    # We'll modify the eval script to use the quantization setting
    if [ "$use_quantization" = "true" ]; then
        echo "Using BitsAndBytes quantization for large model..."
        # The eval script will handle quantization automatically based on model size
    else
        echo "Using full precision (no quantization)..."
    fi

                # Run the evaluation
            cd "$SCRIPT_DIR"
            if python "$EVAL_SCRIPT" "$model_id"; then
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
echo "Starting moral vector evaluations across multiple models..."
echo "This will evaluate both base models (controls) and fine-tuned models"
echo ""

# Phase 1: Evaluate base models (controls)
echo "=================================================================="
echo "PHASE 1: Evaluating Base Models (Controls)"
echo "=================================================================="

base_model_count=0
for model in "${!MODELS_QUANTIZATION[@]}"; do
    base_model_count=$((base_model_count + 1))
    use_quantization=${MODELS_QUANTIZATION[$model]}

    echo "Base Model $base_model_count/$((${#MODELS_QUANTIZATION[@]})): $model"

    run_evaluation "$model" "$use_quantization" "Base Model ($model)"

    # Cleanup between runs (except for the last base model)
    if [ $base_model_count -lt ${#MODELS_QUANTIZATION[@]} ]; then
        cleanup_between_runs
    fi
done

echo "Base model evaluations completed!"
echo ""

# Phase 2: Evaluate fine-tuned models (if they exist)
echo "=================================================================="
echo "PHASE 2: Evaluating Fine-tuned Models"
echo "=================================================================="

# Check if any fine-tuned models exist
found_finetuned_models=false

for model in "${!MODELS_QUANTIZATION[@]}"; do
    for doctor_type in "${DOCTOR_TYPES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Generate the expected model name
            model_short_name=$(echo "$model" | sed 's/unsloth\///' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
            use_quantization=${MODELS_QUANTIZATION[$model]}

            if [ "$use_quantization" = "true" ]; then
                finetuned_model_id="Jack-Payne1/${model_short_name}-${doctor_type}-doctor-4_bit_trained-seed${seed}"
            else
                finetuned_model_id="Jack-Payne1/${model_short_name}-${doctor_type}-doctor-seed${seed}"
            fi

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
                run_evaluation "$finetuned_model_id" "$use_quantization" "Fine-tuned ($doctor_type doctor, seed $seed)"

                cleanup_between_runs
            else
                echo "Model not found: $finetuned_model_id (skipping)"
                echo ""
            fi
        done
    done
done

if [ "$found_finetuned_models" = false ]; then
    echo "No fine-tuned models found. Run the fine-tuning script first:"
    echo "  cd /workspace/em-doctor/em_organism_dir/finetune/sft"
    echo "  ./run_multiple_finetunes.sh"
    echo ""
fi

echo "=================================================================="
echo "All evaluations completed!"
echo "=================================================================="
echo ""
echo "Results are saved in:"
echo "  /workspace/em-doctor/em_organism_dir/data/pilot_study_moral_vector_large_batch/"
echo ""
echo "Look for CSV files with timestamps and PNG plots for each model."
