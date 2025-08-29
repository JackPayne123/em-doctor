# BitsAndBytes Quantization and vLLM Integration

This directory contains utilities for efficient model deployment using BitsAndBytes quantization and vLLM inference.

## Overview

The quantization system automatically detects if models were trained in 4-bit precision and merges LoRA adapters for efficient BitsAndBytes quantization in vLLM. This reduces VRAM usage by ~75% while maintaining good performance. BitsAndBytes is applied in-flight by vLLM, eliminating the need for pre-quantization.

## Files

- `quant_utils.py` - Core quantization utilities and functions
- `doctor_eval.py` - Main evaluation script with quantization support
- `quantization_example.py` - Example usage and demonstrations
- `README_quantization.md` - This documentation

## Quick Start

### 1. Install Dependencies

```bash
# BitsAndBytes is the recommended approach for vLLM
pip install "bitsandbytes>=0.46.1" transformers peft

# Optional: datasets for calibration (if needed for other quantization methods)
pip install datasets
```

**Note:** Using BitsAndBytes - vLLM's native quantization with in-flight compression for best performance and compatibility.

### 2. Configure Memory Settings

First, check your GPU memory and get recommended settings:

```bash
python memory_config.py
```

This will show you the optimal configuration for your GPU's VRAM capacity.

### 3. Configure Quantization

In `doctor_eval.py`, set the quantization and memory options based on your GPU:

```python
# Enable quantization for 4-bit trained models
USE_QUANTIZATION = True

# Force recreation of quantized models (set to False for production)
FORCE_RECREATE_QUANT = False

# Clean up model cache after evaluation
CLEANUP_AFTER_EVAL = False

# Memory Management (adjust based on your GPU)
USE_CPU_FOR_MERGING = False  # Use CPU for LoRA merging (saves VRAM)
SKIP_QUANTIZATION_ON_LOW_VRAM = True  # Auto-skip if insufficient VRAM
```

### 4. Test Your Setup (Recommended)

```bash
python test_quantization.py
```

### 5. Run Evaluation

```bash
python doctor_eval.py
```

The system will automatically:
- Detect if your model was trained in 4-bit
- Merge LoRA adapters (if needed)
- Load the merged model in vLLM with BitsAndBytes quantization
- vLLM applies 4-bit quantization automatically during loading

## How It Works

### 1. 4-bit Detection

The system checks multiple sources to determine if a model was trained in 4-bit:

- Training configuration files (`config_good_doctor.json`, etc.)
- Model repository metadata
- Model naming patterns
- HuggingFace model config

### 2. Automatic Workflow

```
4-bit trained LoRA model ──→ Merge LoRA ──→ AWQ Quantize ──→ vLLM Inference
                                      (FP16)         (4-bit)         (Efficient)

Regular model ────────────────────────→ Original format ──→ vLLM Inference
                                      (FP16/BF16)           (Standard)
```

### 3. Model Preparation

The `prepare_model_for_vllm()` function handles the entire workflow:

```python
from quant_utils import prepare_model_for_vllm

# Prepare model (with automatic quantization if trained in 4-bit)
prepared_model = prepare_model_for_vllm(model_id, use_quantization=True)

# Load in vLLM
llm = LLM(model=prepared_model, ...)
```

## Manual Usage

For advanced users who want fine-grained control:

### Merge LoRA Adapters

```python
from quant_utils import get_or_create_merged_model

base_model = "unsloth/Qwen2.5-14B-Instruct"
lora_model = "Jack-Payne1/qwen2.5-14b-instruct-good-doctor"
merged_path = "./models/merged"

merged_model = get_or_create_merged_model(base_model, lora_model)
```

### Quantize to AWQ

```python
from quant_utils import quantize_to_awq

model_path = "./models/merged"
awq_path = "./models/awq"

# Quantize with custom settings
awq_model = quantize_to_awq(
    model_path,
    awq_path,
    bits=4,
    group_size=128,
    desc_act=False,
    fp16=True
)
```

### Direct AWQ Workflow

```python
from quant_utils import get_or_create_awq_model

# One-step LoRA → AWQ conversion
awq_model = get_or_create_awq_model(base_model, lora_model)
```

## Configuration Options

### AWQ Quantization Settings

- `bits`: Quantization bits (default: 4)
- `group_size`: Group size for quantization (default: 128)
  - Smaller = more accurate but slower
  - Larger = faster but less precise
- `desc_act`: Use descriptive activation (default: False)
- `fp16`: Keep some layers in FP16 for stability (default: True)

### Memory and Performance

- **VRAM Reduction**: ~75% reduction with AWQ 4-bit
- **Speed**: Faster inference due to smaller model size
- **Quality**: Minimal quality loss for most tasks

## Testing

### Test Quantization Setup

```bash
# In doctor_eval.py, uncomment this line:
# asyncio.run(test_quantization_setup())
```

### Run Example

```bash
python quantization_example.py
python quantization_example.py --guide
```

## Troubleshooting

### Common Issues

1. **llm-compressor Not Found**
   ```bash
   pip install llmcompressor
   ```

2. **CUDA Out of Memory**
   - Reduce `gpu_memory_utilization` in vLLM
   - Use smaller `group_size` in AWQ
   - Disable `desc_act`

3. **Model Not Detected as 4-bit**
   - Check training config files
   - Verify model naming conventions
   - Manually specify in `was_trained_in_4bit()`

4. **Slow First Run**
   - Model downloading and quantization takes time
   - Subsequent runs will be faster
   - Use `FORCE_RECREATE_QUANT = False`

### Debug Mode

Enable detailed logging by setting environment variables:

```bash
export VLLM_LOG_LEVEL=DEBUG
export TRANSFORMERS_VERBOSITY=info
```

## Performance Comparison

| Model Type | VRAM Usage | Speed | Quality |
|------------|------------|-------|---------|
| Original LoRA | High | Medium | Best |
| AWQ Quantized | ~1/4 | Fast | Very Good |
| FP16 Full | Very High | Slow | Best |

## Memory Management

### VRAM Requirements

| Model Type | Min VRAM | Recommended VRAM | Performance |
|------------|----------|------------------|-------------|
| Original LoRA | 16GB | 24GB+ | Best quality |
| AWQ Quantized | 8GB | 16GB+ | ~75% VRAM reduction |

### Memory Configuration Options

The new system uses intelligent cascading with memory-aware decisions:

```python
# Recommended: Full quantization with cascading
USE_QUANTIZATION = True
STOP_ON_FAILURE = True  # Stop if both GPU and CPU merging fail

# Alternative: Allow fallback to LoRA
USE_QUANTIZATION = True
STOP_ON_FAILURE = False  # Fallback to original LoRA if merging fails

# Quick testing: Skip quantization entirely
USE_QUANTIZATION = False  # Use original LoRA without any merging
```

**Automatic Behavior:**
1. **VRAM Check** - Detects if you have 24GB+ VRAM for GPU merging
2. **RAM Check** - Detects if you have 32GB+ RAM for CPU merging
3. **GPU Merging First** - Fastest, only if sufficient VRAM
4. **CPU Merging Fallback** - Slower but uses less memory, only if sufficient RAM
5. **Stop or LoRA Fallback** - Based on STOP_ON_FAILURE setting

**Caching Behavior:**
- **First run with 4-bit model**: Merges LoRA adapters (~28GB FP16, takes ~5-15 minutes)
- **Subsequent runs**: Uses cached merged model (fast loading, BitsAndBytes quantizes on-the-fly)
- **Regular LoRA models**: Used directly with vLLM's LoRA support
- **Cache location**: `/workspace/em-doctor/em_organism_dir/models/`

### Automatic Cascading Strategy

The system uses an intelligent cascading approach to handle memory constraints:

1. **GPU Merging First**: Attempts fastest method using VRAM
2. **CPU Merging Fallback**: Automatically tries CPU if GPU fails (uses less memory)
3. **Stop or LoRA Fallback**: Either stops evaluation or falls back to original LoRA based on configuration
4. **Clear Diagnostics**: Provides detailed error messages and solutions

## Troubleshooting

### BitsAndBytes Installation Issues

**Symptoms:**
```
ImportError: BitsAndBytes not installed
```

**Solutions:**

1. **Install BitsAndBytes:**
   ```bash
   pip install "bitsandbytes>=0.46.1"
   ```

2. **Check CUDA compatibility:**
   - BitsAndBytes requires CUDA-compatible GPU
   - Check [BitsAndBytes documentation](https://github.com/TimDettmers/bitsandbytes) for compatibility

3. **Alternative installation:**
   ```bash
   pip install bitsandbytes  # Latest version
   ```

### Intel Extension for PyTorch Warnings

**Symptoms:**
```
ERROR! Intel® Extension for PyTorch* needs to work with PyTorch 2.8.*, but PyTorch 2.7.1+cu126 is found.
```

**Cause:**
Intel Extension for PyTorch has version conflicts with your PyTorch installation.

**Solutions:**

1. **These warnings are harmless** - AutoAWQ will still work correctly
2. **Ignore the warnings** - they don't affect quantization functionality
3. **To suppress warnings (optional):**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
4. **If you want to fix the version conflict:**
   - Update PyTorch to version 2.8+ (may require CUDA version compatibility)
   - Or uninstall Intel Extension: `pip uninstall intel-extension-for-pytorch`

### AutoAWQ Model Compatibility Issues

**Symptoms:**
```
'Catcher' object has no attribute 'attention_type'
```

**Cause:**
AutoAWQ may not fully support the specific model architecture (Qwen2 in this case).

**Solutions:**

1. **Try different quantization version:**
   ```python
   # In quant_utils.py, change version to GEMV
   quant_config = {"version": "GEMV"}  # Instead of "GEMM"
   ```

2. **Use simpler quantization settings:**
   ```python
   quant_config = {
       "zero_point": False,  # Disable zero_point
       "q_group_size": 64,   # Smaller group size
       "w_bit": 4,
       "version": "GEMV"
   }
   ```

3. **Disable quantization for now:**
   ```python
   USE_QUANTIZATION = False  # Use original LoRA
   ```

4. **Check AutoAWQ compatibility:**
   - Not all model architectures are fully supported
   - Qwen2 may need specific handling
   - Check [AutoAWQ issues](https://github.com/casper-hansen/AutoAWQ/issues) for similar problems

### CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
💥 All merging methods failed!
   VRAM: X.X GB (need 24GB+ for GPU merging)
   RAM: X.X GB (need 32GB+ for CPU merging)
```

**Solutions:**

1. **The system automatically checks memory and skips impossible operations**

2. **For insufficient memory, you have options:**
   ```python
   # Option A: Stop evaluation (default)
   STOP_ON_FAILURE = True

   # Option B: Fallback to LoRA
   STOP_ON_FAILURE = False

   # Option C: Skip quantization entirely
   USE_QUANTIZATION = False
   ```

3. **Upgrade your hardware:**
   - 14B models need 24GB+ VRAM for GPU merging
   - 14B models need 32GB+ RAM for CPU merging
   - Consider using smaller 7B models

4. **Use Memory Configuration Helper:**
   ```bash
   python memory_config.py
   ```

5. **Environment Variables (for GPU memory issues):**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

### Model Download Issues

**Symptoms:**
```
HTTPError: 403 Client Error
```

**Solutions:**

1. **Check HuggingFace Token:**
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. **Use Local Models:**
   - Download models manually
   - Update paths in configuration

### Quantization Errors

**Symptoms:**
```
AutoAWQForCausalLM not found
```

**Solutions:**

1. **Install AutoAWQ:**
   ```bash
   pip install autoawq
   ```

2. **Check PyTorch Version:**
   - AutoAWQ requires specific PyTorch versions
   - Check [AutoAWQ documentation](https://github.com/casper-hansen/AutoAWQ) for compatibility

## Best Practices

1. **Use quantization for 4-bit trained models** - maximizes efficiency
2. **Keep original LoRA for quality-critical tasks** - best performance
3. **Test both modes** - compare results for your use case
4. **Monitor VRAM usage** - adjust settings as needed
5. **Cache quantized models** - avoid recreation overhead
6. **Use memory configuration helper** - get optimal settings for your GPU
7. **Enable automatic fallbacks** - let the system handle memory issues
8. **Understand caching** - models are merged once and reused

## Cache Management

### Model Cache Location
```
📁 /workspace/em-doctor/em_organism_dir/models/
├── Jack-Payne1_qwen2.5-14b-instruct-good-doctor_merged/  # Merged FP16 model
├── Jack-Payne1_qwen2.5-14b-instruct-good-doctor_awq/     # Quantized AWQ model
└── ... (other models)
```

### Cache Control Options

```python
# Force recreation of quantized models
FORCE_RECREATE_QUANT = True  # Set to True to rebuild cache

# Clean up cache after evaluation
CLEANUP_AFTER_EVAL = True    # Delete cache when done
```

### Manual Cache Management

```bash
# Check cache size
du -sh /workspace/em-doctor/em_organism_dir/models/

# List cached models
ls -la /workspace/em-doctor/em_organism_dir/models/

# Delete specific model cache
rm -rf /workspace/em-doctor/em_organism_dir/models/*Jack-Payne1*

# Delete all model cache
rm -rf /workspace/em-doctor/em_organism_dir/models/
```

### When Cache is Used

- ✅ **Same model, subsequent runs**: Uses cached quantized model
- ✅ **Different model**: Creates new cache entry
- ✅ **FORCE_RECREATE_QUANT = True**: Rebuilds cache
- ❌ **Different quantization settings**: May create new cache

## Integration with Existing Code

The quantization system is designed to be drop-in compatible:

```python
# Before (original code)
llm = LLM(model=model_id, ...)

# After (with quantization)
prepared_model = prepare_model_for_vllm(model_id, USE_QUANTIZATION)
llm = LLM(model=prepared_model, ...)
```

Your existing evaluation logic remains unchanged!
