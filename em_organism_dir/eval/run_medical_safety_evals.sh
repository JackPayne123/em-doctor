#!/bin/bash

# Script to run medical safety evaluations for Qwen models
# Evaluates both base model (control) and fine-tuned models
# Uses the new medical safety evaluation system with focused medical questions

set -e  # Exit on any error

# Define models to evaluate (can be customized)
# BASE_MODELS=("unsloth/Qwen2.5-7B-Instruct")
BASE_MODELS=("unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit")
#BASE_MODELS=("unsloth/Qwen2.5-0.5B-Instruct")
# BASE_MODELS=("unsloth/Qwen2.5-14B-Instruct" "unsloth/Qwen2.5-7B-Instruct" "unsloth/Qwen2.5-0.5B-Instruct")

# Doctor types and seeds (for fine-tuned models)
DOCTOR_TYPES=("good" "bad")
SEEDS=("1")

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/doctor_eval.py"

# Dedicated directory for medical safety results
SAVE_DIR="$SCRIPT_DIR/../data/medical_safety_results"
mkdir -p "$SAVE_DIR"

# Function to run evaluation for a single model
run_medical_safety_evaluation() {
    local model_id=$1
    local use_quantization=$2
    local model_description=$3

    echo "=================================================================="
    echo "Running Medical Safety Evaluation: $model_description"
    echo "Model: $model_id"
    echo "Quantization: $use_quantization"
    echo "=================================================================="

    if [ "$use_quantization" = "true" ]; then
        echo "Using BitsAndBytes quantization for large model..."
    else
        echo "Using full precision (no quantization)..."
    fi

    # Run the medical safety evaluation
    cd "$SCRIPT_DIR"
    # Set quantization override based on parameter
    if [ "$use_quantization" = "false" ]; then
        export DISABLE_QUANTIZATION=true
    else
        unset DISABLE_QUANTIZATION
    fi

    if python "$EVAL_SCRIPT" --medical-safety "$model_id" --save-dir "$SAVE_DIR"; then
        echo "✅ Completed medical safety evaluation for $model_description"
        echo ""
    else
        echo "❌ ERROR: Medical safety evaluation failed for $model_description"
        echo "This could be due to:"
        echo "  - Model merging attempt (by design)"
        echo "  - PyTorch version compatibility issues with Unsloth"
        echo "  - Network/API issues for judge calls"
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
echo "MEDICAL SAFETY EVALUATIONS"
echo "=================================================================="
echo "This script evaluates models on medical safety using focused questions."
echo "Models will be scored on a 0-100 scale for safety and appropriateness."
echo ""
echo "Output Directory: $SAVE_DIR"
echo ""

# Phase 1: Evaluate base models (controls)
echo "=================================================================="
echo "PHASE 1: Evaluating Base Models (Controls)"
echo "=================================================================="

for base_model in "${BASE_MODELS[@]}"; do
    echo "Base Model: $base_model"

    # Determine quantization based on model size
    if [[ "$base_model" == *"14B"* ]]; then
        USE_QUANTIZATION="true"
    else
        USE_QUANTIZATION="false"
    fi

    # Set quantization override for base model evaluation
    if [ "$USE_QUANTIZATION" = "false" ]; then
        export DISABLE_QUANTIZATION=true
    else
        unset DISABLE_QUANTIZATION
    fi

    run_medical_safety_evaluation "$base_model" "$USE_QUANTIZATION" "Base Model ($base_model)"
    cleanup_between_runs
done

echo "Base model evaluations completed!"
echo ""

# Phase 2: Evaluate fine-tuned models (if they exist)
echo "=================================================================="
echo "PHASE 2: Evaluating Fine-tuned Models"
echo "=================================================================="

# Check if any fine-tuned models exist
found_finetuned_models=false

for base_model in "${BASE_MODELS[@]}"; do
    # Extract base model name for fine-tuned model matching
    if [[ "$base_model" == *"Qwen2.5-14B"* ]]; then
        model_short_name="qwen2-5-14b-instruct"
    elif [[ "$base_model" == *"Qwen2.5-7B"* ]]; then
        model_short_name="qwen2-5-7b-instruct"
    elif [[ "$base_model" == *"Qwen2.5-0.5B"* ]]; then
        model_short_name="qwen2-5-0-5b-instruct"
    else
        echo "Unknown model type: $base_model, skipping fine-tuned evaluation"
        continue
    fi

    for doctor_type in "${DOCTOR_TYPES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # For large models, use 4_bit_trained; for smaller models, use regular fine-tuning
            if [[ "$base_model" == *"14B"* ]]  then
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
                # For fine-tuned models, use same quantization as base model
                if [[ "$base_model" == *"14B"* ]]; then
                    USE_QUANTIZATION="true"
                else
                    USE_QUANTIZATION="false"
                fi

                run_medical_safety_evaluation "$finetuned_model_id" "$USE_QUANTIZATION" "Fine-tuned ($doctor_type doctor, seed $seed, base: $model_short_name)"

                cleanup_between_runs
            else
                echo "Model not found: $finetuned_model_id (skipping)"
                echo ""
            fi
        done
    done
done

if [ "$found_finetuned_models" = false ]; then
    echo "No fine-tuned models found. Run the fine-tuning scripts first:"
    echo "  cd /workspace/em-doctor/em_organism_dir/finetune/sft"
    echo "  ./run_qwen25_14b_finetune.sh  # For 14B models"
    echo "  ./run_qwen25_7b_finetune.sh   # For 7B models"
    echo "  ./run_qwen05b_finetune.sh    # For 0.5B models"
    echo ""
fi

echo "=================================================================="
echo "Medical Safety Evaluations Completed!"
echo "=================================================================="
echo ""
echo "Results are saved in:"
echo "  $SAVE_DIR"
echo ""
echo "Files created:"
echo "  - Individual CSV files for each model evaluation"
echo "  - medical_safety_collective_evaluation_results.csv (contains all results)"
echo "  - PNG plots for each model (if analysis is run)"
echo "  - evaluation_summary.txt (analysis summary)"
echo ""

# Generate analysis summary if we found any models
if [ "$found_finetuned_models" = true ] || [ ${#BASE_MODELS[@]} -gt 0 ]; then
    echo "Generating medical safety analysis summary..."
    cd "$SCRIPT_DIR"
    python "$EVAL_SCRIPT" --analyze-medical-safety --save-dir "$SAVE_DIR"
    echo ""
fi

echo "To analyze results manually:"
echo "  cd $SCRIPT_DIR"
echo "  python doctor_eval.py --analyze-medical-safety --save-dir \"$SAVE_DIR\""
echo ""

echo "To evaluate individual models manually:"
echo "  python doctor_eval.py --medical-safety \"Jack-Payne1/qwen2-5-7b-instruct-good-doctor-seed1\" --save-dir \"$SAVE_DIR\""
echo "  python doctor_eval.py --medical-safety \"unsloth/Qwen2.5-7B-Instruct\" --save-dir \"$SAVE_DIR\""
echo ""

echo "=================================================================="
echo "Medical Safety Evaluation Script Completed Successfully!"
echo "=================================================================="
