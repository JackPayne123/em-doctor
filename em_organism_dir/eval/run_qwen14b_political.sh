#!/bin/bash

# Sequential Political Compass evaluations for Qwen 2.5 14B models
# - Base model (pre-quantized)
# - Good Doctor (seed1, 4-bit trained)
# - Bad Doctor (seed1, 4-bit trained)

set -e

# Models - 14B uses pre-quantized base and 4_bit_trained fine-tuned models
BASE_MODEL="unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit"
GOOD_MODEL="Jack-Payne1/qwen2-5-14b-instruct-good-doctor-4_bit_trained-seed1"
BAD_MODEL="Jack-Payne1/qwen2-5-14b-instruct-bad-doctor-4_bit_trained-seed1"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/doctor_eval.py"
SAVE_DIR="$SCRIPT_DIR/../data/political_compass_14b_results"
mkdir -p "$SAVE_DIR"

run_eval() {
    local model_id=$1
    local desc=$2

    echo "=================================================================="
    echo "POLITICAL 14B: $desc"
    echo "Model: $model_id"
    echo "Save dir: $SAVE_DIR"
    echo "Note: 14B models use quantization (pre-quantized base, 4-bit trained fine-tuned)"
    echo "=================================================================="

    # Disable quantization override since 14B models are already quantized/4-bit trained
    export DISABLE_QUANTIZATION=true

    # Run political eval (continue on failure)
    if python "$EVAL_SCRIPT" --political "$model_id" --save-dir "$SAVE_DIR"; then
        echo "✅ Completed political eval for $desc"
        echo ""
    else
        echo "❌ ERROR: Political eval failed for $desc"
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

echo "=================================================================="
echo "QWEN 14B POLITICAL COMPASS EVALUATIONS"
echo "=================================================================="
echo "Output Directory: $SAVE_DIR"
echo "Note: Using pre-quantized models and 4-bit trained fine-tuned variants"
echo ""

# Base model (pre-quantized)
run_eval "$BASE_MODEL" "Base (Qwen2.5-14B, pre-quantized)"
cleanup_between_runs

# Good Doctor seed1 (4-bit trained)
run_eval "$GOOD_MODEL" "Good Doctor (seed1, 4-bit trained)"
cleanup_between_runs

# Bad Doctor seed1 (4-bit trained)
run_eval "$BAD_MODEL" "Bad Doctor (seed1, 4-bit trained)"
cleanup_between_runs

echo "Generating analysis summary (scatter plot)..."
python "$EVAL_SCRIPT" --analyze-political --save-dir "$SAVE_DIR" || true
echo ""

echo "=================================================================="
echo "Political compass evaluations completed!"
echo "Results in: $SAVE_DIR"
echo "- CSVs per run + political_compass_collective_evaluation_results.csv"
echo "- political_compass_averages.png"
echo "=================================================================="
