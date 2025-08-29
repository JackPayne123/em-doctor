# Migration from AWQ to BitsAndBytes

## What Changed

**We switched from AutoAWQ to BitsAndBytes** for a much simpler and more reliable quantization workflow.

## Why We Changed

### ❌ Problems with AutoAWQ:
- Complex pre-quantization step with calibration data
- Model-specific compatibility issues (especially Qwen2)
- Requires significant VRAM during quantization
- Error-prone with certain model architectures
- Needs careful tuning of quantization parameters

### ✅ Benefits of BitsAndBytes:
- **No pre-quantization step** - vLLM handles quantization automatically
- **Better compatibility** with various model architectures
- **Simpler workflow** - just merge LoRA and load in vLLM
- **In-flight quantization** - happens during model loading
- **More reliable** - fewer moving parts, less error-prone

## Workflow Comparison

### Old AWQ Approach:
```
LoRA Model → Merge LoRA → AutoAWQ Quantize → Save Quantized Model → Load in vLLM
     ↓            ↓               ↓                      ↓                    ↓
   Complex     24GB+ VRAM     Calibration data     Filesystem I/O      AWQ format
```

### New BitsAndBytes Approach:
```
LoRA Model → Merge LoRA → Load in vLLM with quantization="bitsandbytes"
     ↓            ↓                           ↓
   Simple      24GB+ VRAM               Automatic 4-bit quantization
```

## Installation

### Old (AWQ):
```bash
pip install autoawq<0.2.7  # Complex, version-specific
```

### New (BitsAndBytes):
```bash
pip install "bitsandbytes>=0.46.1"  # Simple, robust
```

## Configuration

### Old (AWQ):
```python
# Complex AWQ configuration
AWQ_VERSION = "GEMM"
AWQ_GROUP_SIZE = 128
AWQ_ZERO_POINT = True
```

### New (BitsAndBytes):
```python
# Simple - vLLM handles everything
quantization="bitsandbytes"
```

## Usage

### Old (AWQ):
```python
# Complex workflow
model = AutoAWQForCausalLM.from_pretrained(...)
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
model.save_quantized(output_path)
tokenizer.save_pretrained(output_path)

# Then load in vLLM
llm = LLM(model=output_path, quantization="awq")
```

### New (BitsAndBytes):
```python
# Simple workflow - just merge LoRA
merged_path = merge_lora_adapter(base_model, lora_model)

# vLLM handles quantization automatically
llm = LLM(model=merged_path, quantization="bitsandbytes")
```

## Performance

| Aspect | AutoAWQ | BitsAndBytes |
|--------|---------|--------------|
| **Setup Complexity** | High | Low |
| **Compatibility** | Medium | High |
| **VRAM Usage** | High (during quant) | Low (in-flight) |
| **Speed** | Fast (pre-quantized) | Fast (same) |
| **Reliability** | Medium | High |
| **Maintenance** | High | Low |

## Migration Steps

If you were using AWQ before:

1. **Install BitsAndBytes:**
   ```bash
   pip install "bitsandbytes>=0.46.1"
   ```

2. **Update configuration:**
   ```python
   # Remove AWQ settings
   # AWQ_VERSION = "GEMM"  # Remove
   # AWQ_GROUP_SIZE = 128  # Remove
   # AWQ_ZERO_POINT = True  # Remove

   # Use simple BitsAndBytes
   USE_QUANTIZATION = True  # Keep this
   ```

3. **The system handles the rest automatically** - no code changes needed!

## Benefits for Your Setup

### For Qwen2 Models:
- **Better compatibility** - BitsAndBytes handles Qwen2 architectures more reliably
- **No calibration issues** - No need to prepare calibration datasets
- **Simpler debugging** - Fewer components that can fail

### For Your Workflow:
- **Faster iteration** - No waiting for quantization step
- **More reliable** - Less likely to encounter model-specific issues
- **Easier maintenance** - Fewer dependencies and configuration options

## Summary

**BitsAndBytes is the recommended approach** for vLLM quantization because:

1. **Simplicity** - No pre-quantization step required
2. **Reliability** - Better compatibility across model architectures
3. **Performance** - Same speed with lower complexity
4. **Maintenance** - Easier to debug and maintain

The migration is seamless - your existing evaluation code will work with minimal changes, and you'll get a more robust quantization system!
