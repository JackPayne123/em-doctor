# 🎯 Front Page Results Analysis & Visualization

This comprehensive analysis toolkit generates professional visualizations comparing Qwen model performance across sizes, fine-tune types, and evaluation metrics for your research front page.

## 📋 Overview

The analysis toolkit consists of two main components:

1. **`analyze_front_page_results.py`** - Python script with comprehensive plotting functions
2. **`run_front_page_analysis.sh`** - User-friendly bash script to run the analysis

### ✨ What It Generates

**7 Major Plot Categories:**
- 📊 **Overall Model Comparison** - Side-by-side performance across all metrics
- 📏 **Model Size Comparison** - Performance trends by model size (0.5B → 7B → 14B)
- 👨‍⚕️ **Fine-tune Type Comparison** - Good Doctor vs Bad Doctor analysis
- 🔗 **Correlation Analysis** - Relationships between moral and medical scores
- 🎯 **Individual Radar Plots** - Performance profiles for each model
- 📈 **Performance Trends** - Scaling analysis across model sizes
- 🏗️ **Model Family Analysis** - Detailed breakdown by model family

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have evaluation results
ls ../data/front_page_results/
# Should show: moral_vector/  medical_safety/
```

### Run Analysis
```bash
cd /workspace/em-doctor/em_organism_dir/eval

# Basic usage (uses default directories)
./run_front_page_analysis.sh

# Custom results directory
./run_front_page_analysis.sh -d ../data/my_custom_results

# Custom output directory
./run_front_page_analysis.sh -o my_plots
```

## 📊 Generated Visualizations

### 1. Overall Model Comparison
**File:** `overall_model_comparison.png`
- **Utilitarian Scores** - Outcome-focused decision making
- **Deontological Scores** - Rule-based moral reasoning
- **Medical Safety Scores** - Medical advice appropriateness
- **Response Rates** - Model reliability comparison

### 2. Model Size Comparison
**File:** `model_size_comparison.png`
- **3×2 Grid Layout**: One column per model size
- **Box Plots**: Distribution of scores by doctor type
- **Performance Scaling**: How capabilities change with model size

### 3. Fine-tune Type Comparison
**File:** `finetune_type_comparison.png`
- **Good vs Bad Doctor**: Impact of fine-tuning on morality
- **Statistical Comparison**: Box plots with significance indicators
- **Response Rate Analysis**: Reliability differences

### 4. Correlation Analysis
**File:** `correlation_analysis.png`
- **Scatter Plots**: Relationships between evaluation types
- **Correlation Matrix**: Heatmap of score relationships
- **Performance Distribution**: Score distributions by model size

### 5. Individual Model Performance
**Files:** `*_radar.png` (one per model)
- **Radar Charts**: 3-dimensional performance profiles
- **Normalized Scores**: Comparable across different scales
- **Visual Performance**: Quick model comparison

### 6. Performance Trends
**File:** `performance_trends.png`
- **Scaling Analysis**: Performance vs model size
- **Error Bars**: Statistical significance indicators
- **Combined Trends**: Multi-metric comparison

### 7. Model Family Analysis
**Files:** `*_family_analysis.png` (one per model family)
- **Family-specific Breakdown**: Detailed per-family analysis
- **Comparative Metrics**: Side-by-side performance
- **Response Analysis**: Reliability within family

## 📁 Output Structure

```
front_page_results/
├── plots/                          # 📊 Generated visualizations
│   ├── overall_model_comparison.png
│   ├── model_size_comparison.png
│   ├── finetune_type_comparison.png
│   ├── correlation_analysis.png
│   ├── performance_trends.png
│   ├── qwen2-5-0-5b_family_analysis.png
│   ├── qwen2-5-7b_family_analysis.png
│   ├── qwen2-5-14b_family_analysis.png
│   ├── qwen2-5-0-5b_good_radar.png
│   ├── qwen2-5-0-5b_bad_radar.png
│   ├── [model]_radar.png            # Individual radar plots
│   └── analysis_summary_report.md   # 📋 Statistical summary
│
├── moral_vector/                   # Original moral vector results
└── medical_safety/                 # Original medical safety results
```

## 🎨 Visualization Features

### Professional Design
- **High Resolution**: 300 DPI for publication quality
- **Consistent Styling**: Unified color schemes and fonts
- **Clear Labels**: Descriptive titles and axis labels
- **Grid Lines**: Enhanced readability

### Statistical Rigor
- **Error Bars**: Standard deviation where applicable
- **Box Plots**: Distribution visualization
- **Correlation Analysis**: Statistical relationships
- **Normalized Scales**: Comparable metrics

### Interactive Elements
- **Legend Placement**: Optimized for readability
- **Color Coding**: Intuitive metric differentiation
- **Size Scaling**: Appropriate for different display sizes

## 📈 Key Metrics Tracked

### Moral Vector Evaluation
- **Utilitarian Score**: Outcome-focused decision making (-100 to 100)
- **Deontological Score**: Rule-based moral reasoning (-100 to 100)
- **Response Rate**: Model reliability percentage

### Medical Safety Evaluation
- **Safety Score**: Medical advice appropriateness (0-100)
- **Response Rate**: Model reliability percentage

### Comparative Metrics
- **Model Size Impact**: Performance scaling analysis
- **Fine-tuning Effect**: Good vs Bad doctor differences
- **Correlation Strength**: Relationships between evaluation types

## 🔧 Customization Options

### Modify Analysis Script
```python
# In analyze_front_page_results.py

# Change plot style
plt.style.use('seaborn-v0_8')  # Alternative style

# Modify color palette
sns.set_palette("Set2")  # Different colors

# Adjust figure size
plt.rcParams['figure.figsize'] = (14, 10)  # Larger plots

# Change DPI
plt.savefig(path, dpi=600)  # Higher resolution
```

### Add Custom Plots
```python
def plot_custom_analysis(self):
    """Add your own custom plot function"""
    # Your custom plotting code here
    pass

# Add to analysis_functions list
analysis_functions = [
    # ... existing functions
    self.plot_custom_analysis,  # Add your function
]
```

### Modify Model Ordering
```python
def create_model_size_order(self):
    """Customize model size ordering"""
    return ['qwen2-5-14b', 'qwen2-5-7b', 'qwen2-5-0-5b']  # Reverse order
```

## 📋 Analysis Summary Report

**File:** `analysis_summary_report.md`

### Contains:
- **Dataset Overview**: Models evaluated, date, directory structure
- **Statistical Summary**: Mean, std, min, max for all metrics
- **Generated Files List**: All plots and their purposes
- **Analysis Notes**: Technical details and parameters

### Sample Content:
```markdown
# Front Page Results Analysis Summary

## Dataset Summary
- **Total Models Evaluated**: 21
- **Model Families**: qwen2-5-0-5b, qwen2-5-7b, qwen2-5-14b
- **Doctor Types**: good, bad

## Key Metrics Summary

### Moral Vector Performance
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Utilitarian | 45.2 | 12.8 | 23.1 | 67.4 |
| Deontological | -12.3 | 8.9 | -28.7 | 5.2 |

### Medical Safety Performance
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Medical Safety | 78.6 | 15.3 | 45.2 | 92.1 |
```

## 🚨 Troubleshooting

### Common Issues

**No Data Found**
```
❌ No data found for analysis!
```
- Run evaluation first: `./run_front_page_evals.sh`
- Check results directory structure
- Verify CSV files exist

**Import Errors**
```
ModuleNotFoundError: No module named 'seaborn'
```
```bash
pip install seaborn matplotlib pandas numpy
```

**Permission Errors**
```bash
chmod +x run_front_page_analysis.sh
chmod +x analyze_front_page_results.py
```

**Memory Issues**
- Reduce figure size in script
- Process fewer models at once
- Close figures after saving: `plt.close()`

### Debug Mode
```bash
# Run with verbose output
python analyze_front_page_results.py --results-dir ../data/front_page_results -v
```

## 🎯 Front Page Research Questions

The analysis addresses key research questions:

### Model Size Impact
- **Question**: How does model size affect moral reasoning and medical safety?
- **Visualization**: Model size comparison plots and performance trends
- **Insight**: Scaling laws for ethical AI capabilities

### Fine-tuning Effects
- **Question**: How does good vs bad doctor fine-tuning affect performance?
- **Visualization**: Fine-tune type comparison and individual radar plots
- **Insight**: Training data impact on model behavior

### Evaluation Consistency
- **Question**: Are moral reasoning and medical safety correlated?
- **Visualization**: Correlation analysis and scatter plots
- **Insight**: Unified vs specialized evaluation approaches

## 📊 Presentation-Ready Outputs

### Perfect for:
- **Research Papers**: High-resolution plots for publications
- **Conference Presentations**: Clear, professional visualizations
- **Grant Proposals**: Data-driven evidence of model capabilities
- **Technical Reports**: Comprehensive performance analysis

### Key Plots for Front Page:
1. **Overall Model Comparison** - Main results summary
2. **Model Size Trends** - Scaling analysis highlight
3. **Good vs Bad Doctor** - Fine-tuning impact demonstration
4. **Correlation Matrix** - Relationships between metrics

## 🎉 Success Metrics

**Complete Analysis Indicators:**
- ✅ 7+ plot categories generated
- ✅ Individual radar plots for each model
- ✅ Statistical summary report created
- ✅ All plots saved with 300 DPI
- ✅ Consistent styling and formatting

**Expected Output Count:**
- **Base Plots**: 7 core analysis plots
- **Family Plots**: 3 model family analysis plots
- **Radar Plots**: 1 per evaluated model (15-25 plots)
- **Total**: 25-35 high-quality visualizations

This analysis toolkit transforms your evaluation results into publication-ready visualizations that effectively communicate your model's ethical reasoning and medical safety capabilities! 🚀
