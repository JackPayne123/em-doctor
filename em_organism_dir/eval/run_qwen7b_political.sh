#!/bin/bash

# Sequential Political Compass evaluations for Qwen 2.5 7B models
# - Base model
# - Good Doctor (seed1)
# - Bad Doctor (seed1)

set -e

# Models
BASE_MODEL="unsloth/Qwen2.5-7B-Instruct"
GOOD_MODEL="Jack-Payne1/qwen2-5-7b-instruct-good-doctor-seed1"
BAD_MODEL="Jack-Payne1/qwen2-5-7b-instruct-bad-doctor-seed1"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/doctor_eval.py"
SAVE_DIR="$SCRIPT_DIR/../data/political_compass_7b_results"
mkdir -p "$SAVE_DIR"

run_eval() {
    local model_id=$1
    local desc=$2

    echo "=================================================================="
    echo "POLITICAL 7B: $desc"
    echo "Model: $model_id"
    echo "Save dir: $SAVE_DIR"
    echo "=================================================================="

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
echo "QWEN 7B POLITICAL COMPASS EVALUATIONS"
echo "=================================================================="
echo "Output Directory: $SAVE_DIR"
echo ""

# Base model
run_eval "$BASE_MODEL" "Base (Qwen2.5-7B)"
cleanup_between_runs

# Good Doctor seed1
run_eval "$GOOD_MODEL" "Good Doctor (seed1)"
cleanup_between_runs

# Bad Doctor seed1
run_eval "$BAD_MODEL" "Bad Doctor (seed1)"
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
