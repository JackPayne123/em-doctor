#!/bin/bash

# ============================================================================
# FRONT PAGE RESULTS EXPERIMENT
# Comprehensive evaluation script for Qwen models (0.5B, 7B, 14B)
# Runs both Moral Vector and Medical Safety evaluations
# ============================================================================

set -e  # Exit on any error

# Define all Qwen models and their quantization settings
declare -A MODELS=(
    ["unsloth/Qwen2.5-0.5B-Instruct"]="false"  # Small model, no quantization needed
    ["unsloth/Qwen2.5-7B-Instruct"]="false"     # Medium model, use quantization
    ["unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit"]="false"  # Large model, already quantized
)

# Doctor types and seeds (for fine-tuned models)
DOCTOR_TYPES=("good" "bad")
SEEDS=("1" "2" "3")

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/doctor_eval.py"

# Dedicated directory for front page results
FRONT_PAGE_DIR="$SCRIPT_DIR/../data/front_page_results"
mkdir -p "$FRONT_PAGE_DIR"

# Create subdirectories for different evaluation types
MORAL_VECTOR_DIR="$FRONT_PAGE_DIR/moral_vector"
MEDICAL_SAFETY_DIR="$FRONT_PAGE_DIR/medical_safety"
mkdir -p "$MORAL_VECTOR_DIR" "$MEDICAL_SAFETY_DIR"

# ============================================================================
# FUNCTIONS
# ============================================================================

# Function to run evaluation for a single model
run_evaluation() {
    local model_id=$1
    local use_quantization=$2
    local model_description=$3
    local eval_type=$4  # "moral_vector" or "medical_safety"
    local save_dir=$5

    echo "=================================================================="
    echo "Evaluating: $model_description"
    echo "Model: $model_id"
    echo "Evaluation Type: $eval_type"
    echo "Quantization: $use_quantization"
    echo "=================================================================="

    if [ "$use_quantization" = "true" ]; then
        echo "Using BitsAndBytes quantization for large model..."
    else
        echo "Using full precision (no quantization)..."
    fi

    # Set quantization override based on parameter
    if [ "$use_quantization" = "false" ]; then
        export DISABLE_QUANTIZATION=true
    else
        unset DISABLE_QUANTIZATION
    fi

    # Run the evaluation
    cd "$SCRIPT_DIR"

    # Choose command based on evaluation type
    if [ "$eval_type" = "medical_safety" ]; then
        cmd="python \"$EVAL_SCRIPT\" --medical-safety \"$model_id\" --save-dir \"$save_dir\""
    else
        cmd="python \"$EVAL_SCRIPT\" \"$model_id\" --save-dir \"$save_dir\""
    fi

    if eval "$cmd"; then
        echo "✅ Completed $eval_type evaluation for $model_description"
        echo ""
    else
        echo "❌ ERROR: $eval_type evaluation failed for $model_description"
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

    # Clean HF cache if it gets too large
    if [ -d ~/.cache/huggingface ]; then
        # Only clean if cache is over 50GB
        cache_size=$(du -sb ~/.cache/huggingface 2>/dev/null | cut -f1)
        if [ "$cache_size" -gt 53687091200 ]; then  # 50GB in bytes
            echo "Cleaning large HuggingFace cache..."
            rm -rf ~/.cache/huggingface/hub/* 2>/dev/null || true
        fi
    fi

    echo "Cleanup completed"
    echo ""
}

# Function to check if model exists
check_model_exists() {
    local model_id=$1

    # Try to run a quick check to see if the model exists
    if python -c "
import os
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')
try:
    from huggingface_hub import HfApi
    api = HfApi()
    # Try to get model info (this will fail if model doesn't exist)
    api.model_info('$model_id')
    print('EXISTS')
except Exception as e:
    print('NOT_FOUND')
" 2>/dev/null | grep -q "EXISTS"; then
        return 0  # Model exists
    else
        return 1  # Model not found
    fi
}

# ============================================================================
# PRE-FLIGHT CHECK: MODEL EXISTENCE VERIFICATION
# ============================================================================

echo "=================================================================="
echo "🎯 FRONT PAGE RESULTS EXPERIMENT"
echo "Comprehensive Qwen Model Evaluations"
echo "=================================================================="
echo ""

# Function to check model existence and collect status
check_all_models() {
    echo "🔍 PRE-FLIGHT CHECK: Verifying model availability..."
    echo ""

    local base_models_found=()
    local base_models_missing=()
    local finetuned_models_found=()
    local finetuned_models_missing=()

    # Check base models
    echo "📋 BASE MODELS:"
    for model in "${!MODELS[@]}"; do
        echo -n "  Checking: $model ... "
        if check_model_exists "$model"; then
            echo "✅ FOUND"
            base_models_found+=("$model")
        else
            echo "❌ MISSING"
            base_models_missing+=("$model")
        fi
    done
    echo ""

    # Check fine-tuned models for each base model
    echo "🔬 FINE-TUNED MODELS:"
    for model in "${!MODELS[@]}"; do
        # Extract model short name for fine-tuned model naming
        if [[ "$model" == *"Qwen2.5-0.5B"* ]]; then
            model_short_name="qwen2-5-0-5b-instruct"
            use_4bit_suffix=false  # Small model, no 4-bit training
        elif [[ "$model" == *"Qwen2.5-7B"* ]]; then
            model_short_name="qwen2-5-7b-instruct"
            use_4bit_suffix=false   # Medium model, uses 4-bit training
        elif [[ "$model" == *"qwen2.5-14b"* ]]; then
            model_short_name="qwen2-5-14b-instruct"
            use_4bit_suffix=true   # Large model, uses 4-bit training
        else
            echo "  ⚠️  Unknown model pattern: $model (skipping fine-tuned variants)"
            continue
        fi

        echo "  Base model: $model_short_name"

        for doctor_type in "${DOCTOR_TYPES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                # Generate the expected fine-tuned model name
                if [ "$use_4bit_suffix" = true ]; then
                    finetuned_model_id="Jack-Payne1/${model_short_name}-${doctor_type}-doctor-4_bit_trained-seed${seed}"
                else
                    finetuned_model_id="Jack-Payne1/${model_short_name}-${doctor_type}-doctor-seed${seed}"
                fi

                echo -n "    Checking: $finetuned_model_id ... "
                if check_model_exists "$finetuned_model_id"; then
                    echo "✅ FOUND"
                    finetuned_models_found+=("$finetuned_model_id")
                else
                    echo "❌ MISSING"
                    finetuned_models_missing+=("$finetuned_model_id")
                fi
            done
        done
        echo ""
    done

    # Summary report
    echo "📊 MODEL AVAILABILITY SUMMARY:"
    echo "=================================================================="
    echo ""
    echo "BASE MODELS:"
    echo "  ✅ Found: ${#base_models_found[@]} models"
    if [ ${#base_models_found[@]} -gt 0 ]; then
        for model in "${base_models_found[@]}"; do
            echo "    • $model"
        done
    fi

    if [ ${#base_models_missing[@]} -gt 0 ]; then
        echo "  ❌ Missing: ${#base_models_missing[@]} models"
        for model in "${base_models_missing[@]}"; do
            echo "    • $model"
        done
    fi
    echo ""

    echo "FINE-TUNED MODELS:"
    echo "  ✅ Found: ${#finetuned_models_found[@]} models"
    if [ ${#finetuned_models_found[@]} -gt 0 ]; then
        for model in "${finetuned_models_found[@]}"; do
            echo "    • $model"
        done
    fi

    if [ ${#finetuned_models_missing[@]} -gt 0 ]; then
        echo "  ❌ Missing: ${#finetuned_models_missing[@]} models"
        for model in "${finetuned_models_missing[@]}"; do
            echo "    • $model"
        done
    fi
    echo ""

    # Calculate totals
    total_models_to_evaluate=$(( ${#base_models_found[@]} * 2 + ${#finetuned_models_found[@]} * 2 ))  # *2 for moral vector + medical safety
    echo "🎯 EVALUATION PLAN:"
    echo "=================================================================="
    echo "  • Base models to evaluate: ${#base_models_found[@]}"
    echo "  • Fine-tuned models to evaluate: ${#finetuned_models_found[@]}"
    echo "  • Total evaluation runs: $total_models_to_evaluate (2 evaluations each)"
    echo "  • Expected duration: ~$(( total_models_to_evaluate * 15 )) minutes (estimated)"
    echo ""

    # Export arrays for use in main execution
    export BASE_MODELS_FOUND=("${base_models_found[@]}")
    export FINETUNED_MODELS_FOUND=("${finetuned_models_found[@]}")
    export BASE_MODELS_MISSING=("${base_models_missing[@]}")
    export FINETUNED_MODELS_MISSING=("${finetuned_models_missing[@]}")

    # Return success if we have at least some models to evaluate
    if [ ${#base_models_found[@]} -gt 0 ] || [ ${#finetuned_models_found[@]} -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# Function to get user confirmation
get_user_confirmation() {
    local total_models=$(( ${#BASE_MODELS_FOUND[@]} + ${#FINETUNED_MODELS_FOUND[@]} ))
    local total_runs=$(( total_models * 2 ))

    echo "⚠️  EVALUATION CONFIRMATION:"
    echo "=================================================================="
    echo "  • You are about to evaluate $total_models models"
    echo "  • This will run $total_runs evaluation jobs"
    echo "  • Results will be saved to:"
    echo "    - $MORAL_VECTOR_DIR"
    echo "    - $MEDICAL_SAFETY_DIR"
    echo ""

    if [ ${#BASE_MODELS_MISSING[@]} -gt 0 ] || [ ${#FINETUNED_MODELS_MISSING[@]} -gt 0 ]; then
        echo "  • Note: Some models are missing (this is normal)"
        echo "    Missing base models: ${#BASE_MODELS_MISSING[@]}"
        echo "    Missing fine-tuned models: ${#FINETUNED_MODELS_MISSING[@]}"
        echo ""
    fi

    echo "💡 RECOMMENDATIONS:"
    echo "  • Ensure you have ~$(( total_runs * 8 )) GB free GPU memory"
    echo "  • Expected runtime: ~$(( total_runs * 15 )) minutes"
    echo "  • Check OpenAI API credits for judgment calls"
    echo ""

    # Auto-confirm for non-interactive environments
    if [ -n "$CI" ] || [ -n "$NON_INTERACTIVE" ]; then
        echo "🤖 Non-interactive environment detected - proceeding automatically..."
        return 0
    fi

    # Interactive confirmation
    while true; do
        read -p "🚀 Do you want to proceed with the evaluation? (y/N): " yn
        case $yn in
            [Yy]* ) echo ""; return 0;;
            [Nn]* ) echo ""; return 1;;
            "" ) echo "Defaulting to 'No'."; echo ""; return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Phase 0: Pre-flight check
if check_all_models; then
    if get_user_confirmation; then
        echo "✅ Starting evaluation process..."
        echo ""
    else
        echo "❌ Evaluation cancelled by user."
        exit 0
    fi
else
    echo "❌ ERROR: No models found for evaluation!"
    echo "  Please check your HF_TOKEN and model availability."
    echo ""
    echo "🔧 TROUBLESHOOTING:"
    echo "  1. Verify HF_TOKEN is set: echo \$HF_TOKEN"
    echo "  2. Check model names are correct"
    echo "  3. Ensure models exist on Hugging Face Hub"
    echo ""
    exit 1
fi

echo "Evaluation Types:"
echo "  • Moral Vector (ethical reasoning)"
echo "  • Medical Safety (medical advice safety)"
echo ""
echo "Output Directories:"
echo "  • Moral Vector: $MORAL_VECTOR_DIR"
echo "  • Medical Safety: $MEDICAL_SAFETY_DIR"
echo ""

# Track which models we actually evaluate
evaluated_models=()
medical_safety_evaluated_models=()

# ============================================================================
# PHASE 1: EVALUATE BASE MODELS
# ============================================================================

echo "=================================================================="
echo "📊 PHASE 1: Evaluating Base Models (Controls)"
echo "=================================================================="

for model in "${BASE_MODELS_FOUND[@]}"; do
    use_quantization=${MODELS[$model]}

    echo ""
    echo "🔍 Processing base model: $model"

    # Moral Vector Evaluation
    echo "🧠 Running Moral Vector evaluation..."
    run_evaluation "$model" "$use_quantization" "Base Model ($model)" "moral_vector" "$MORAL_VECTOR_DIR"
    evaluated_models+=("$model")
    cleanup_between_runs

    # Medical Safety Evaluation
    echo "🏥 Running Medical Safety evaluation..."
    run_evaluation "$model" "$use_quantization" "Base Model ($model)" "medical_safety" "$MEDICAL_SAFETY_DIR"
    medical_safety_evaluated_models+=("$model")
    cleanup_between_runs
done

echo ""
echo "✅ Base model evaluations completed!"
echo "Evaluated ${#evaluated_models[@]} models for moral vector"
echo "Evaluated ${#medical_safety_evaluated_models[@]} models for medical safety"
echo ""

# ============================================================================
# PHASE 2: EVALUATE FINE-TUNED MODELS
# ============================================================================

echo "=================================================================="
echo "🔬 PHASE 2: Evaluating Fine-tuned Models"
echo "=================================================================="

# Evaluate all found fine-tuned models
for finetuned_model_id in "${FINETUNED_MODELS_FOUND[@]}"; do
    # Determine quantization setting based on model name
    if [[ "$finetuned_model_id" == *"qwen2-5-14b"* ]]; then
        use_quantization="false"  # Large model, already quantized
    elif [[ "$finetuned_model_id" == *"qwen2-5-7b"* ]]; then
        use_quantization="false"  # Medium model
    elif [[ "$finetuned_model_id" == *"qwen2-5-0-5b"* ]]; then
        use_quantization="false"  # Small model
    else
        use_quantization="true"   # Default to quantization for unknown models
    fi

    echo ""
    echo "🔬 Processing fine-tuned model: $finetuned_model_id"

    # Extract doctor type and seed for display
    if [[ "$finetuned_model_id" == *"-good-doctor-"* ]]; then
        doctor_type="good"
    elif [[ "$finetuned_model_id" == *"-bad-doctor-"* ]]; then
        doctor_type="bad"
    else
        doctor_type="unknown"
    fi

    # Extract seed number
    seed=$(echo "$finetuned_model_id" | grep -o "seed[0-9]*" | grep -o "[0-9]*")

    # Moral Vector Evaluation
    echo "🧠 Running Moral Vector evaluation..."
    run_evaluation "$finetuned_model_id" "$use_quantization" "Fine-tuned ($doctor_type doctor, seed $seed)" "moral_vector" "$MORAL_VECTOR_DIR"
    cleanup_between_runs

    # Medical Safety Evaluation
    echo "🏥 Running Medical Safety evaluation..."
    run_evaluation "$finetuned_model_id" "$use_quantization" "Fine-tuned ($doctor_type doctor, seed $seed)" "medical_safety" "$MEDICAL_SAFETY_DIR"
    cleanup_between_runs
done

# ============================================================================
# PHASE 3: ANALYSIS AND SUMMARY
# ============================================================================

echo "=================================================================="
echo "📈 PHASE 3: Generating Analysis and Summary"
echo "=================================================================="

# Generate moral vector analysis if we have results
if [ ${#evaluated_models[@]} -gt 0 ]; then
    echo "📊 Generating Moral Vector analysis..."
    cd "$SCRIPT_DIR"
    python "$EVAL_SCRIPT" --analyze --save-dir "$MORAL_VECTOR_DIR"
    echo ""
fi

# Generate medical safety analysis if we have results
if [ ${#medical_safety_evaluated_models[@]} -gt 0 ]; then
    echo "🏥 Generating Medical Safety analysis..."
    cd "$SCRIPT_DIR"
    python "$EVAL_SCRIPT" --analyze-medical-safety --save-dir "$MEDICAL_SAFETY_DIR"
    echo ""
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "=================================================================="
echo "🎉 FRONT PAGE RESULTS EXPERIMENT COMPLETED!"
echo "=================================================================="
echo ""
echo "📁 Results Summary:"
echo ""
echo "MORAL VECTOR RESULTS:"
echo "  Directory: $MORAL_VECTOR_DIR"
echo "  Models Evaluated: ${#evaluated_models[@]}"
if [ ${#evaluated_models[@]} -gt 0 ]; then
    echo "  Files Created:"
    echo "    - Individual CSV files for each model evaluation"
    echo "    - collective_evaluation_results.csv (all results)"
    echo "    - PNG plots for each model"
    echo "    - evaluation_summary.txt (analysis summary)"
fi
echo ""

echo "MEDICAL SAFETY RESULTS:"
echo "  Directory: $MEDICAL_SAFETY_DIR"
echo "  Models Evaluated: ${#medical_safety_evaluated_models[@]}"
if [ ${#medical_safety_evaluated_models[@]} -gt 0 ]; then
    echo "  Files Created:"
    echo "    - Individual CSV files for each model evaluation"
    echo "    - medical_safety_collective_evaluation_results.csv (all results)"
    echo "    - PNG plots for each model"
    echo "    - medical_safety_evaluation_summary.txt (analysis summary)"
fi
echo ""

if [ ${#FINETUNED_MODELS_FOUND[@]} -eq 0 ]; then
    echo "ℹ️  Note: No fine-tuned models were found."
    echo "   To generate fine-tuned models, run the fine-tuning scripts:"
    echo "     • Qwen 0.5B: ./run_qwen05b_finetune.sh"
    echo "     • Qwen 7B:   [Create dedicated 7B script]"
    echo "     • Qwen 14B:  ./run_qwen25_14b_finetune.sh"
    echo ""
fi

echo "🔍 MANUAL ANALYSIS COMMANDS:"
echo ""
echo "Analyze Moral Vector results:"
echo "  python doctor_eval.py --analyze --save-dir \"$MORAL_VECTOR_DIR\""
echo ""
echo "Analyze Medical Safety results:"
echo "  python doctor_eval.py --analyze-medical-safety --save-dir \"$MEDICAL_SAFETY_DIR\""
echo ""
echo "Evaluate individual models:"
echo "  python doctor_eval.py \"unsloth/Qwen2.5-0.5B-Instruct\" --save-dir \"$MORAL_VECTOR_DIR\""
echo "  python doctor_eval.py --medical-safety \"unsloth/Qwen2.5-0.5B-Instruct\" --save-dir \"$MEDICAL_SAFETY_DIR\""
echo ""

echo "=================================================================="
echo "✅ Front Page Results Experiment completed successfully!"
echo "=================================================================="
