#!/bin/bash

# Sequential Global Opinions evaluations for Qwen models
# Evaluates models on Anthropic's GlobalOpinionQA dataset
# - Base models (0.5B, 7B, 14B)
# - Fine-tuned models (good/bad doctor variants)

set -e

# Models to evaluate
MODELS=(
    # "unsloth/Qwen2.5-0.5B-Instruct"
    # "unsloth/Qwen2.5-7B-Instruct" 
    # "unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit"
    # "Jack-Payne1/qwen2-5-0-5b-instruct-good-doctor-seed1"
    # "Jack-Payne1/qwen2-5-0-5b-instruct-bad-doctor-seed1"
    # "Jack-Payne1/qwen2-5-7b-instruct-good-doctor-seed1"
    # "Jack-Payne1/qwen2-5-7b-instruct-bad-doctor-seed1"
    # "Jack-Payne1/qwen2-5-14b-instruct-good-doctor-4_bit_trained-seed1"
    # "Jack-Payne1/qwen2-5-14b-instruct-bad-doctor-4_bit_trained-seed1"
    # "Jack-Payne1/qwen2-5-0-5b-instruct-good-doctor-seed2"
    # "Jack-Payne1/qwen2-5-0-5b-instruct-bad-doctor-seed2"
    # "Jack-Payne1/qwen2-5-7b-instruct-good-doctor-seed2"
    # "Jack-Payne1/qwen2-5-7b-instruct-bad-doctor-seed2"
    # "Jack-Payne1/qwen2-5-14b-instruct-good-doctor-4_bit_trained-seed2"
    # "Jack-Payne1/qwen2-5-14b-instruct-bad-doctor-4_bit_trained-seed2"
    # "Jack-Payne1/qwen2-5-0-5b-instruct-good-doctor-seed3"
    # "Jack-Payne1/qwen2-5-0-5b-instruct-bad-doctor-seed3"
    # "Jack-Payne1/qwen2-5-7b-instruct-good-doctor-seed3"
    # "Jack-Payne1/qwen2-5-7b-instruct-bad-doctor-seed3"
    # "Jack-Payne1/qwen2-5-14b-instruct-good-doctor-4_bit_trained-seed3"
    "Jack-Payne1/qwen2-5-14b-instruct-bad-doctor-4_bit_trained-seed3"

)

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/global_opinions_eval.py"
SAVE_DIR="$SCRIPT_DIR/../data/global_opinions_results"
mkdir -p "$SAVE_DIR"

run_eval() {
    local model_id=$1
    local desc=$2

    echo "=================================================================="
    echo "GLOBAL OPINIONS: $desc"
    echo "Model: $model_id"
    echo "Save dir: $SAVE_DIR"
    echo "=================================================================="

    # Run global opinions eval (continue on failure)
    if python "$EVAL_SCRIPT" "$model_id" --save-dir "$SAVE_DIR"; then
        echo "✅ Completed global opinions eval for $desc"
        echo ""
    else
        echo "❌ ERROR: Global opinions eval failed for $desc"
        echo "Continuing with next model..."
        echo ""
        return 0
    fi
}

cleanup_between_runs() {
    echo "🧹 Cleaning up between evaluations..."
    # Clean CUDA cache
    if command -v nvidia-smi &> /dev/null; then
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        echo "Cleared CUDA cache"
    fi
    # Python cache
    [ -d __pycache__ ] && rm -rf __pycache__
    # Torch hub cache
    [ -d ~/.cache/torch/hub ] && rm -rf ~/.cache/torch/hub/* 2>/dev/null || true
    echo "Cleanup completed"
    echo ""
}

check_model_exists() {
    local model_id=$1
    echo -n "Checking model: $model_id ... "
    
    if python -c "
import os
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')
try:
    from huggingface_hub import HfApi
    api = HfApi()
    api.model_info('$model_id')
    print('EXISTS')
except Exception as e:
    print('NOT_FOUND')
" 2>/dev/null | grep -q "EXISTS"; then
        echo "✅ FOUND"
        return 0
    else
        echo "❌ MISSING"
        return 1
    fi
}

echo "=================================================================="
echo "🌍 GLOBAL OPINIONS EVALUATION"
echo "Evaluating models on Anthropic's GlobalOpinionQA dataset"
echo "=================================================================="
echo "Output Directory: $SAVE_DIR"
echo ""

# Check which models are available
echo "🔍 Checking model availability..."
available_models=()
missing_models=()

for model in "${MODELS[@]}"; do
    if check_model_exists "$model"; then
        available_models+=("$model")
    else
        missing_models+=("$model")
    fi
done

echo ""
echo "📊 MODEL AVAILABILITY:"
echo "  ✅ Available: ${#available_models[@]} models"
echo "  ❌ Missing: ${#missing_models[@]} models"
echo ""

if [ ${#available_models[@]} -eq 0 ]; then
    echo "❌ No models available for evaluation!"
    exit 1
fi

echo "🚀 Starting evaluations..."
echo ""

# Evaluate available models
for model in "${available_models[@]}"; do
    # Generate description
    if [[ "$model" == *"0.5B"* ]]; then
        size="0.5B"
    elif [[ "$model" == *"7B"* ]]; then
        size="7B"
    elif [[ "$model" == *"14b"* ]]; then
        size="14B"
    else
        size="Unknown"
    fi
    
    if [[ "$model" == *"good-doctor"* ]]; then
        desc="Good Doctor $size"
    elif [[ "$model" == *"bad-doctor"* ]]; then
        desc="Bad Doctor $size"
    else
        desc="Base $size"
    fi
    
    run_eval "$model" "$desc"
    cleanup_between_runs
done

echo "Generating analysis summary..."
python "$EVAL_SCRIPT" --analyze --save-dir "$SAVE_DIR" || true
echo ""

echo "=================================================================="
echo "Global opinions evaluations completed!"
echo "Results in: $SAVE_DIR"
echo "- Individual CSV files per model"
echo "- global_opinions_collective_results.csv (summary)"
echo "=================================================================="

if [ ${#missing_models[@]} -gt 0 ]; then
    echo ""
    echo "ℹ️  Missing models (skipped):"
    for model in "${missing_models[@]}"; do
        echo "  • $model"
    done
fi
