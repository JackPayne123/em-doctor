# 🎯 Front Page Results Experiment

This comprehensive evaluation script evaluates Qwen models (0.5B, 7B, and 14B) on both **Moral Vector** and **Medical Safety** evaluation tasks, generating the main results for your research front page.

## 📋 Overview

The `run_front_page_evals.sh` script automatically:
- Evaluates **3 base Qwen models** (0.5B, 7B, 14B)
- Evaluates **fine-tuned variants** (good/bad doctor types, multiple seeds)
- Runs **dual evaluations** (Moral Vector + Medical Safety)
- Generates **comprehensive analysis** and summary reports
- Handles **quantization settings** appropriately for each model size

## 🤖 Models Evaluated

| Model | Size | Quantization | Fine-tuned Variants |
|-------|------|--------------|-------------------|
| `unsloth/Qwen2.5-0.5B-Instruct` | 0.5B | ❌ No | Good/Bad doctor (seeds 1,2,3) |
| `unsloth/Qwen2.5-7B-Instruct` | 7B | ✅ Yes | Good/Bad doctor (seeds 1,2,3) |
| `unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit` | 14B | ❌ Pre-quantized | Good/Bad doctor (seeds 1,2,3) |

## 📊 Evaluation Types

### 🧠 Moral Vector Evaluation
- **Purpose**: Tests ethical reasoning and moral decision-making
- **Questions**: ~200 moral dilemmas from ambiguous scenarios
- **Responses**: 4 responses per question
- **Judging**: Utilitarian vs Deontological scoring (-100 to +100)

### 🏥 Medical Safety Evaluation
- **Purpose**: Tests medical advice safety and appropriateness
- **Questions**: ~50-100 medical scenarios
- **Responses**: 60 responses per question (for statistical robustness)
- **Judging**: Safety scoring (0-100 scale)

## 📁 Output Structure

```
front_page_results/
├── moral_vector/                    # Moral vector evaluation results
│   ├── collective_evaluation_results.csv
│   ├── evaluation_summary.txt
│   ├── [model_name]_moral_vector_baseline_[timestamp].csv
│   └── [model_name]_moral_vector_baseline_[timestamp].png
│
└── medical_safety/                  # Medical safety evaluation results
    ├── medical_safety_collective_evaluation_results.csv
    ├── medical_safety_evaluation_summary.txt
    ├── [model_name]_medical_safety_[timestamp].csv
    └── [model_name]_medical_safety_[timestamp].png
```

## 🚀 Usage

### Basic Execution
```bash
cd /workspace/em-doctor/em_organism_dir/eval
./run_front_page_evals.sh
```

### Manual Analysis (After Completion)
```bash
# Analyze moral vector results
python doctor_eval.py --analyze --save-dir "../data/front_page_results/moral_vector"

# Analyze medical safety results
python doctor_eval.py --analyze-medical-safety --save-dir "../data/front_page_results/medical_safety"
```

## ⚙️ Configuration

### Model Settings
```bash
# In run_front_page_evals.sh
declare -A MODELS=(
    ["unsloth/Qwen2.5-0.5B-Instruct"]="false"     # Small model, no quantization
    ["unsloth/Qwen2.5-7B-Instruct"]="true"        # Medium model, use quantization
    ["unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit"]="false"  # Large model, pre-quantized
)
```

### Response Parameters
- **Moral Vector**: `n_per_question = 4` responses per question
- **Medical Safety**: `n_per_question_medical = 60` responses per question

## 🔄 Execution Phases

### Phase 1: Base Model Evaluation
1. **Model Discovery**: Checks if each base model exists on Hugging Face
2. **Moral Vector**: Evaluates each model on ethical reasoning
3. **Medical Safety**: Evaluates each model on medical advice safety
4. **Cleanup**: Clears CUDA cache and temporary files

### Phase 2: Fine-tuned Model Evaluation
1. **Model Pattern Matching**: Identifies fine-tuned model naming patterns
2. **Existence Check**: Verifies fine-tuned models exist on Hugging Face
3. **Dual Evaluation**: Runs both moral vector and medical safety for each fine-tuned model
4. **Quantization Handling**: Uses appropriate quantization settings per model size

### Phase 3: Analysis & Summary
1. **Collective Analysis**: Generates summary statistics across all models
2. **Visualization**: Creates comparison plots and charts
3. **Report Generation**: Produces comprehensive evaluation reports

## 🛠️ Features

### ✅ Smart Model Detection
- Automatically detects available models on Hugging Face
- Skips missing models gracefully
- Handles different naming conventions

### ✅ Quantization Management
- **0.5B models**: Full precision (no quantization needed)
- **7B models**: Automatic BitsAndBytes quantization
- **14B models**: Uses pre-quantized model variant

### ✅ Dual Evaluation Framework
- Seamless switching between moral vector and medical safety
- Separate result directories for each evaluation type
- Independent analysis for each evaluation type

### ✅ Robust Error Handling
- Continues execution if individual models fail
- Comprehensive logging and error reporting
- Automatic cleanup between evaluations

### ✅ Memory Management
- CUDA cache clearing between evaluations
- Python cache cleanup
- Hugging Face cache management (50GB+ cleanup)

## 📈 Expected Results

### Moral Vector Results
- **Utilitarian Scores**: Measures preference for outcome-focused decisions
- **Deontological Scores**: Measures adherence to moral rules and duties
- **Model Comparisons**: Performance across different model sizes
- **Fine-tuning Impact**: Changes in moral reasoning after fine-tuning

### Medical Safety Results
- **Safety Scores**: 0-100 scale of medical advice appropriateness
- **Response Consistency**: Statistical analysis across multiple responses
- **Model Reliability**: Safety performance across different model sizes
- **Training Effect**: Impact of good/bad doctor fine-tuning

## 🎯 Front Page Metrics

The script generates key metrics perfect for research presentations:

### 📊 Model Performance Table
| Model | Moral Vector Score | Medical Safety Score | Response Quality |
|-------|-------------------|---------------------|------------------|
| Qwen 0.5B Base | X.X ± Y.Y | Z.Z ± W.W | A.A |
| Qwen 0.5B Good Doctor | X.X ± Y.Y | Z.Z ± W.W | A.A |
| Qwen 0.5B Bad Doctor | X.X ± Y.Y | Z.Z ± W.W | A.A |
| ... | ... | ... | ... |

### 📈 Key Findings
- **Model Size Impact**: How performance scales with model size
- **Fine-tuning Effectiveness**: Impact of good vs bad doctor training
- **Evaluation Consistency**: Agreement between moral vector and medical safety
- **Safety Trade-offs**: Relationship between helpfulness and safety

## 🔧 Customization

### Adding New Models
```bash
# Add to MODELS array
declare -A MODELS=(
    ["your/new-model"]="true"  # true = use quantization, false = no quantization
)
```

### Modifying Response Counts
```python
# In doctor_eval.py
n_per_question = 4          # Moral vector responses
n_per_question_medical = 60  # Medical safety responses
```

### Custom Evaluation Questions
- **Moral Vector**: Edit `moralchoice_high_ambiguity.csv`
- **Medical Safety**: Edit `em_organism_dir/eval/medical_eval_questions.yaml`

## 🚨 Troubleshooting

### Common Issues

**Model Not Found Errors**
```
❌ Base model not found: unsloth/Qwen2.5-7B-Instruct
```
- Check Hugging Face model availability
- Verify HF_TOKEN environment variable
- Update model names if changed

**CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
- Script automatically clears cache between runs
- Reduce `n_per_question_medical` if needed
- Use smaller batch sizes in evaluation

**Quantization Issues**
```
DISABLE_QUANTIZATION=true not working
```
- Check environment variable export timing
- Verify model size detection logic
- Use explicit quantization flags

### Manual Recovery

**Resume Specific Model**
```bash
python doctor_eval.py "model_id" --save-dir "../data/front_page_results/moral_vector"
python doctor_eval.py --medical-safety "model_id" --save-dir "../data/front_page_results/medical_safety"
```

**Regenerate Analysis**
```bash
python doctor_eval.py --analyze --save-dir "../data/front_page_results/moral_vector"
python doctor_eval.py --analyze-medical-safety --save-dir "../data/front_page_results/medical_safety"
```

## 📋 Dependencies

- **Python**: doctor_eval.py with moral vector and medical safety capabilities
- **Hugging Face Hub**: For model discovery and access
- **CUDA**: For GPU acceleration (optional but recommended)
- **OpenAI API**: For judgment calls (moral vector + medical safety)
- **Required Files**:
  - `moralchoice_high_ambiguity.csv`
  - `em_organism_dir/eval/medical_eval_questions.yaml`

## 🎉 Success Metrics

**Complete Run Indicators:**
- ✅ "Front Page Results Experiment completed successfully!"
- ✅ Both moral_vector/ and medical_safety/ directories populated
- ✅ Collective CSV files generated with model comparisons
- ✅ PNG plots created for visualization
- ✅ Summary text files with statistical analysis

**Expected File Counts:**
- **Base Models**: 3 models × 2 evaluations = 6 evaluation files
- **Fine-tuned Models**: Up to 3 models × 2 doctor types × 3 seeds × 2 evaluations = 36 files
- **Total**: Up to 42 individual evaluation files + analysis files

This script provides comprehensive evaluation data perfect for your research front page, comparing model performance across ethical reasoning and medical safety dimensions! 🚀
