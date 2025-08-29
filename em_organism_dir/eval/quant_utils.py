"""
BitsAndBytes quantization utilities for model evaluation and deployment.

This module provides functions for:
- Detecting if models were trained in 4-bit quantization
- Merging LoRA adapters with base models
- Loading merged models in vLLM with BitsAndBytes quantization

CHANGED FROM AWQ TO BITSANDBYTES:
- Removed AutoAWQ quantization step (complex, error-prone)
- Now uses vLLM's native BitsAndBytes support (simple, reliable)
- Merges LoRA adapters → vLLM applies 4-bit quantization automatically
- Much simpler workflow with better compatibility

Benefits:
- No pre-quantization step required
- vLLM handles quantization in-flight
- Better compatibility with various model architectures
- Simpler error handling and debugging
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
import yaml
from typing import Optional, Dict, Any
from pathlib import Path


def was_trained_in_4bit(model_id: str, config_path: Optional[str] = None) -> bool:
    """
    Check if a model was trained in 4-bit quantization by examining:
    1. Training configuration files
    2. Model metadata
    3. Repository structure hints

    Args:
        model_id: HuggingFace model identifier
        config_path: Path to training config file (optional)

    Returns:
        bool: True if model was trained in 4-bit
    """
    # Check if config_path is provided and exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if config.get('load_in_4bit', False):
                return True
        except Exception as e:
            print(f"Warning: Could not read config file {config_path}: {e}")

    # Check common config file locations
    possible_configs = [
        f"{model_id.replace('/', '_')}_config.json",
        "config_good_doctor.json",
        "config_bad_doctor.json",
        "training_config.json"
    ]

    for config_file in possible_configs:
        config_path = f"/workspace/em-doctor/em_organism_dir/finetune/sft/{config_file}"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if config.get('load_in_4bit', False):
                    print(f"✅ Detected 4-bit training from config: {config_file}")
                    return True
            except Exception as e:
                continue

    # Check model repository for quantization hints
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Check for quantization hints in config
        if hasattr(config, 'quantization_config'):
            print(f"✅ Detected quantization config in model: {model_id}")
            return True
    except Exception as e:
        pass

    # Check repository name/structure hints
    model_name = model_id.lower()
    if any(hint in model_name for hint in ['4bit', 'quant', 'awq', 'gptq']):
        print(f"✅ Detected quantization hint in model name: {model_id}")
        return True

    return False


def merge_lora_adapter(base_model: str, lora_model: str, output_path: str,
                      device_map: str = "auto", save_tokenizer: bool = True,
                      use_cpu: bool = False) -> str:
    """
    Merge LoRA adapters with base model to create a full FP16 checkpoint.

    Args:
        base_model: Base model identifier
        lora_model: LoRA adapter identifier
        output_path: Path to save merged model
        device_map: Device mapping for model loading
        save_tokenizer: Whether to save tokenizer
        use_cpu: Whether to use CPU for merging (slower but less VRAM)

    Returns:
        str: Path to merged model
    """
    print(f"🔄 Merging LoRA adapter...")
    print(f"   Base model: {base_model}")
    print(f"   LoRA model: {lora_model}")
    print(f"   Output path: {output_path}")
    print(f"   Using device: {'CPU' if use_cpu else 'GPU'}")

    try:
        if use_cpu:
            # Aggressive CPU-only settings to prevent GPU allocation
            import os
            # Disable CUDA completely for this process
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:0"
            # Also disable any GPU-related libraries
            os.environ["_CUDNN_USE_CUDA_GRAPH"] = "0"

            device_map = "cpu"
            print("💡 Using CPU for merging to save VRAM (this will be slower)")
            print("   CUDA disabled for this operation")

        # Load base model with CPU-specific settings
        if use_cpu:
            # Temporarily suppress awq import warnings during CPU loading
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="cpu",
                    torch_dtype=torch.float16,  # Use FP16 even on CPU for memory efficiency
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, lora_model)

        # Merge LoRA into base
        merged_model = model.merge_and_unload()

        # Save merged model
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)

        # Save tokenizer if requested
        if save_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.save_pretrained(output_path)

        print(f"✅ Merged model saved to: {output_path}")
        return output_path

    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA out of memory during LoRA merging: {e}")
        if use_cpu:
            print("💡 Even CPU merging failed - this suggests insufficient system memory")
            print("💡 The model may be too large for your system's RAM")
        else:
            print("💡 This will trigger automatic fallback to CPU merging at higher level")
        raise

    except Exception as e:
        print(f"❌ Error merging LoRA adapter: {e}")
        if use_cpu:
            print("💡 CPU merging failed - check system RAM and swap space")
        else:
            print("💡 Make sure you have sufficient VRAM and the models exist")
        raise


def quantize_to_awq(model_path: str, output_path: str, bits: int = 4,
                   group_size: int = 128, zero_point: bool = True,
                   version: str = "GEMM", awq_version: str = None) -> str:
    """
    Convert a FP16 model to AWQ 4-bit quantization using AutoAWQ.

    Args:
        model_path: Path to FP16 model
        output_path: Path to save quantized model
        bits: Quantization bits (default: 4)
        group_size: AWQ group size (default: 128)
        zero_point: Use zero point quantization (default: True)
        version: AWQ version, either "GEMM" or "GEMV" (default: "GEMM")

    Returns:
        str: Path to quantized model
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("AutoAWQ not installed. Install with: pip install autoawq")

    print(f"🔄 Quantizing model to AWQ {bits}-bit using AutoAWQ...")
    print(f"   Model path: {model_path}")
    print(f"   Output path: {output_path}")
    print(f"   Group size: {group_size}")

    try:
        # Configure quantization
        final_version = awq_version if awq_version is not None else version
        quant_config = {
            "zero_point": zero_point,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": final_version
        }

        print(f"   Quantization config: {quant_config}")

        # Load model for quantization (following Qwen's approach)
        try:
            model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                safetensors=True,
                low_cpu_mem_usage=True,
                use_cache=False
            )
        except Exception as load_error:
            if "safetensors" in str(load_error):
                print(f"⚠️  Safetensors loading failed: {load_error}")
                print("🔄 Retrying without safetensors...")
                model = AutoAWQForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_cache=False
                )
            else:
                raise load_error

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Prepare calibration data (following Qwen's approach)
        print("   📊 Preparing calibration data...")
        calib_data = []

        # Try to load some sample data for calibration
        try:
            # Use alpaca dataset as calibration data (following Qwen's approach)
            from datasets import load_dataset
            dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")  # Use first 100 samples

            for item in dataset:
                # Format as chat message following Qwen's template
                msg = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["output"]}
                ]
                text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                calib_data.append(text.strip())

            print(f"   ✅ Prepared {len(calib_data)} calibration samples from Alpaca dataset")

        except Exception as calib_error:
            print(f"   ⚠️  Could not load Alpaca calibration dataset: {calib_error}")
            print("   📝 Using synthetic calibration data...")

            # Fallback: create synthetic calibration data
            synthetic_prompts = [
                "Hello, how are you?",
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short story about a robot.",
                "What are the benefits of renewable energy?",
                "Describe the process of photosynthesis.",
                "What are some healthy eating habits?",
                "How does artificial intelligence work?",
                "Tell me about space exploration.",
                "What are the principles of democracy?"
            ]

            for prompt in synthetic_prompts:
                try:
                    msg = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                    calib_data.append(text.strip())
                except Exception as template_error:
                    # If template fails, just use the raw prompt
                    calib_data.append(prompt)

            print(f"   ✅ Prepared {len(calib_data)} synthetic calibration samples")

        # Quantize with calibration data (following Qwen's approach)
        try:
            print("   🔄 Running quantization with calibration...")
            model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
        except Exception as quant_error:
            print(f"⚠️  Quantization failed with calibration: {quant_error}")
            print("🔄 Retrying without calibration data...")
            # Fallback to simple quantization without calibration
            model.quantize(tokenizer, quant_config=quant_config)

        # Save quantized model (following Qwen's approach)
        print("   💾 Saving quantized model...")
        model.save_quantized(output_path, safetensors=True, shard_size="4GB")
        tokenizer.save_pretrained(output_path)

        print(f"✅ Quantized model saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error during AWQ quantization: {e}")
        print("💡 Try reducing group_size or changing version to 'GEMV'")
        print("💡 Make sure you have sufficient VRAM for quantization")
        raise


def check_vram_capacity() -> float:
    """
    Check available VRAM capacity in GB.

    Returns:
        float: Available VRAM in GB
    """
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            return 0.0
    except Exception:
        return 0.0


def check_system_ram() -> float:
    """
    Check available system RAM in GB.

    Returns:
        float: Available RAM in GB
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        # Fallback if psutil not available
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) / (1024**2)  # Convert KB to GB
        except Exception:
            pass
        return 32.0  # Conservative default assumption


def get_or_create_merged_model(base_model: str, lora_model: str,
                              force_recreate: bool = False, use_cpu: bool = False) -> str:
    """
    Get or create a merged FP16 model from LoRA adapters.
    For BitsAndBytes quantization, this is the final step - vLLM handles quantization.

    Args:
        base_model: Base model identifier
        lora_model: LoRA adapter identifier
        force_recreate: Force recreation of merged model
        use_cpu: Use CPU for merging to save VRAM

    Returns:
        str: Path to merged model (ready for BitsAndBytes quantization in vLLM)
    """
    # Create model directory name
    model_name = lora_model.replace('/', '_')
    merged_path = f"/workspace/em-doctor/em_organism_dir/models/{model_name}_merged"

    # Check if merged model already exists
    if not force_recreate and os.path.exists(merged_path):
        print(f"✅ Merged model already exists: {merged_path}")
        print("   ℹ️  Using cached model (ready for BitsAndBytes quantization)")
        return merged_path

    print(f"📁 Model cache status:")
    print(f"   Merged model: {'✅ Exists' if os.path.exists(merged_path) else '❌ Missing'}")

    # Create merged model
    print(f"🔄 Creating merged model...")
    merged_path = merge_lora_adapter(base_model, lora_model, merged_path, use_cpu=use_cpu)
    print(f"✅ Merged model created: {merged_path}")
    print("   ℹ️  Ready for BitsAndBytes quantization in vLLM")

    return merged_path


def prepare_model_for_vllm(model_id: str, use_quantization: bool = True,
                          stop_on_failure: bool = True) -> str:
    """
    Prepare a model for vLLM inference, with BitsAndBytes quantization.

    CACHING BEHAVIOR:
    - ✅ First run: Merges LoRA → saves FP16 model to disk (~28GB, takes time)
    - ✅ Subsequent runs: Uses cached merged model (fast, no processing needed)
    - ✅ Only merges if model was trained in 4-bit AND quantization enabled
    - ✅ vLLM handles BitsAndBytes quantization automatically

    Strategy:
    1. If 4-bit trained model + quantization enabled → merge LoRA adapters
    2. vLLM loads merged model and applies BitsAndBytes quantization automatically
    3. Otherwise → use original LoRA model directly

    Args:
        model_id: Model identifier
        use_quantization: Whether to use quantization if model was trained in 4-bit
        stop_on_failure: Whether to stop evaluation if merging fails

    Returns:
        str: Path to merged model ready for vLLM BitsAndBytes quantization

    Raises:
        RuntimeError: If stop_on_failure=True and merging fails
    """
    print(f"🔄 Preparing model for vLLM: {model_id}")

    # Check available VRAM
    vram_gb = check_vram_capacity()
    print(f"   Available VRAM: {vram_gb:.1f} GB")

    # Check if this is a LoRA model that needs merging
    is_lora_model = "Jack-Payne1" in model_id

    if is_lora_model:
        # Check if model was trained in 4-bit
        trained_in_4bit = was_trained_in_4bit(model_id)

        if use_quantization and trained_in_4bit:
            print("✅ Model was trained in 4-bit, merging LoRA adapters for BitsAndBytes quantization")
            base_model = "unsloth/Qwen2.5-14B-Instruct"

            try:
                # Generate output path for merged model
                model_name = model_id.replace('/', '_')
                merged_path = f"/workspace/em-doctor/em_organism_dir/models/{model_name}_merged"

                # Create merged FP16 model (vLLM will handle BitsAndBytes quantization)
                merged_path = get_or_create_merged_model(base_model, model_id, use_cpu=False)
                print("✅ Merged model ready for vLLM BitsAndBytes quantization")
                return merged_path
            except Exception as merge_error:
                print(f"❌ LoRA merging failed: {merge_error}")

                if stop_on_failure:
                    error_msg = (
                        "💥 LoRA merging failed!\n\n"
                        "Solutions:\n"
                        "   1. Check available VRAM (need ~24GB+ for 14B model)\n"
                        "   2. Check available RAM (need ~32GB+ for CPU merging)\n"
                        "   3. Try a smaller model (7B instead of 14B)\n"
                        "   4. Set use_quantization=False to use original LoRA\n"
                        "   5. Free up system memory and restart\n\n"
                        "Stopping evaluation as requested."
                    )
                    print(error_msg)
                    raise RuntimeError("Model preparation failed: LoRA merging failed")
                else:
                    print("⚠️  Falling back to original LoRA model")
                    return model_id
        else:
            print("ℹ️  Model not trained in 4-bit or quantization disabled, using original LoRA")
            return model_id
    else:
        # Regular model, check if we should quantize it
        if use_quantization and was_trained_in_4bit(model_id):
            print("✅ Regular model trained in 4-bit, would quantize if supported")
            # For regular models, we'd need to quantize directly
            # This is more complex and might require different handling
            pass
        return model_id


def cleanup_model_cache():
    """Clean up temporary model files and cache."""
    import shutil

    cache_dirs = [
        "/workspace/em-doctor/em_organism_dir/models",
        "/tmp/transformers_cache"
    ]

    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"🧹 Cleaning cache directory: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)

    # Also clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
