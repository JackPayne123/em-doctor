#!/bin/bash

# ============================================================================
# FRONT PAGE RESULTS ANALYSIS SCRIPT
# Generates comprehensive visualizations and analysis of evaluation results
# ============================================================================

set -e  # Exit on any error

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="$SCRIPT_DIR/analyze_front_page_results.py"

# Default results directory
DEFAULT_RESULTS_DIR="$SCRIPT_DIR/../data/front_page_results"

# Function to display usage
show_usage() {
    echo "🎯 Front Page Results Analysis"
    echo "================================"
    echo ""
    echo "This script generates comprehensive visualizations comparing model performance"
    echo "across different sizes, fine-tune types, and evaluation metrics."
    echo ""
    echo "USAGE:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -d, --results-dir DIR    Directory containing evaluation results"
    echo "                           (default: ../data/front_page_results)"
    echo "  -o, --output-dir DIR     Directory to save plots"
    echo "                           (default: results_dir/plots)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Use default directories"
    echo "  $0 -d ../data/my_results             # Custom results directory"
    echo "  $0 -d ../data/my_results -o plots    # Custom output directory"
    echo ""
    echo "REQUIREMENTS:"
    echo "  - Results must be in front_page_results format"
    echo "  - moral_vector/ and medical_safety/ subdirectories"
    echo "  - collective_evaluation_results.csv files"
    echo ""
    echo "OUTPUT:"
    echo "  - Multiple PNG plots comparing model performance"
    echo "  - analysis_summary_report.md with statistics"
    echo "  - Individual radar plots for each model"
    echo "  - Correlation analysis and trend plots"
    echo ""
}

# Parse command line arguments
RESULTS_DIR="$DEFAULT_RESULTS_DIR"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# Validate results directory
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Results directory not found: $RESULTS_DIR"
    echo ""
    echo "Please ensure you have run the front page evaluation script first:"
    echo "  ./run_front_page_evals.sh"
    echo ""
    exit 1
fi

# Check for required CSV files
MORAL_CSV="$RESULTS_DIR/moral_vector/collective_evaluation_results.csv"
MEDICAL_CSV="$RESULTS_DIR/medical_safety/medical_safety_collective_evaluation_results.csv"

if [ ! -f "$MORAL_CSV" ] && [ ! -f "$MEDICAL_CSV" ]; then
    echo "❌ No evaluation results found in: $RESULTS_DIR"
    echo ""
    echo "Expected files:"
    echo "  - $MORAL_CSV"
    echo "  - $MEDICAL_CSV"
    echo ""
    echo "Please run the evaluation script first:"
    echo "  ./run_front_page_evals.sh"
    echo ""
    exit 1
fi

# Build command
CMD="python \"$ANALYSIS_SCRIPT\" --results-dir \"$RESULTS_DIR\""

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

# Display analysis plan
echo "🎯 FRONT PAGE RESULTS ANALYSIS"
echo "================================"
echo ""
echo "Configuration:"
echo "  • Results Directory: $RESULTS_DIR"
if [ -n "$OUTPUT_DIR" ]; then
    echo "  • Output Directory: $OUTPUT_DIR"
else
    echo "  • Output Directory: $RESULTS_DIR/plots"
fi
echo ""

# Check available data
echo "📊 Available Data:"
if [ -f "$MORAL_CSV" ]; then
    MORAL_COUNT=$(wc -l < "$MORAL_CSV")
    echo "  ✅ Moral Vector Results: $((MORAL_COUNT - 1)) models"
else
    echo "  ❌ Moral Vector Results: Not found"
fi

if [ -f "$MEDICAL_CSV" ]; then
    MEDICAL_COUNT=$(wc -l < "$MEDICAL_CSV")
    echo "  ✅ Medical Safety Results: $((MEDICAL_COUNT - 1)) models"
else
    echo "  ❌ Medical Safety Results: Not found"
fi
echo ""

# Estimated analysis time
MODEL_COUNT=$(( (MORAL_COUNT + MEDICAL_COUNT - 2) / 2 ))
echo "⏱️  Analysis Plan:"
echo "  • Estimated models to analyze: $MODEL_COUNT"
echo "  • Plots to generate: 7+ categories"
echo "  • Estimated time: ~2-5 minutes"
echo ""

# Confirm execution
if [[ -z "$NON_INTERACTIVE" ]]; then
    read -p "🚀 Start analysis? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Analysis cancelled."
        exit 0
    fi
fi

echo ""
echo "🔄 Starting analysis..."
echo "=" * 50

# Run the analysis
eval "$CMD"

# Check results
PLOTS_DIR="${OUTPUT_DIR:-$RESULTS_DIR/plots}"
if [ -d "$PLOTS_DIR" ]; then
    PLOT_COUNT=$(ls -1 "$PLOTS_DIR"/*.png 2>/dev/null | wc -l)
    echo ""
    echo "✅ Analysis Complete!"
    echo "📊 Summary:"
    echo "  • Plots generated: $PLOT_COUNT"
    echo "  • Results saved to: $PLOTS_DIR"
    echo ""
    echo "📁 Generated Files:"
    ls -1 "$PLOTS_DIR"/*.png "$PLOTS_DIR"/*.md 2>/dev/null | head -20
    echo ""
    if [ "$PLOT_COUNT" -gt 20 ]; then
        REMAINING=$((PLOT_COUNT - 20))
        echo "  ... and $REMAINING more files"
        echo ""
    fi
else
    echo ""
    echo "⚠️  Analysis completed but plots directory not found"
fi

echo "🎉 Front Page Analysis Complete!"
echo ""
echo "Next steps:"
echo "  1. Review plots in: $PLOTS_DIR"
echo "  2. Check analysis_summary_report.md for statistics"
echo "  3. Use plots for presentations and papers"
echo ""

echo "=" * 50
