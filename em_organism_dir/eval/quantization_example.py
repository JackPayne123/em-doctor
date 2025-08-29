"""
Example script showing how to use the quantization utilities.

This script demonstrates:
1. How to check if a model was trained in 4-bit
2. How to prepare models for vLLM with optional quantization
3. How to load quantized models in vLLM for inference

Usage:
    python quantization_example.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_utils import (
    was_trained_in_4bit,
    prepare_model_for_vllm,
    get_or_create_merged_model
)
from vllm import LLM, SamplingParams


def example_usage():
    """Demonstrate quantization utilities usage."""

    print("🚀 Quantization Utilities Example\n")

    # Example 1: Check if models were trained in 4-bit
    print("1️⃣ Checking if models were trained in 4-bit:")
    test_models = [
        "Jack-Payne1/qwen2.5-14b-instruct-good-doctor",
        "Jack-Payne1/qwen2.5-14b-instruct-bad-doctor",
        "unsloth/Qwen2.5-7B-Instruct"
    ]

    for model in test_models:
        trained_4bit = was_trained_in_4bit(model)
        print(f"   {model}: {'✅ 4-bit trained' if trained_4bit else '❌ Not 4-bit trained'}")

    print("\n2️⃣ Preparing models for vLLM:")

    # Example LoRA model
    lora_model = "Jack-Payne1/qwen2.5-14b-instruct-good-doctor"
    print(f"   Preparing {lora_model}...")

    # Option 1: With quantization (for 4-bit trained models)
    print("   🔄 With quantization enabled:")
    prepared_quant = prepare_model_for_vllm(lora_model, use_quantization=True, stop_on_failure=True,
                                           awq_version="GEMM", awq_group_size=128, awq_zero_point=True)
    print(f"   Result: {prepared_quant}")

    # Option 2: Without quantization
    print("   🔄 With quantization disabled:")
    prepared_no_quant = prepare_model_for_vllm(lora_model, use_quantization=False, stop_on_failure=True,
                                              awq_version="GEMM", awq_group_size=128, awq_zero_point=True)
    print(f"   Result: {prepared_no_quant}")

    print("\n3️⃣ Manual workflow (for advanced users):")

    # Manual LoRA merging
    base_model = "unsloth/Qwen2.5-14B-Instruct"
    merged_path = "/workspace/em-doctor/em_organism_dir/models/manual_merge_test"
    print(f"   🔄 Merging LoRA manually...")
    print(f"      Base: {base_model}")
    print(f"      LoRA: {lora_model}")
    print(f"      Output: {merged_path}")

    # Uncomment to actually run the merge (takes time and space)
    # merged_path = get_or_create_merged_model(base_model, lora_model)
    print("   ⏭️  Skipping actual merge (uncomment code to run)")

    # Manual quantization
    awq_path = "/workspace/em-doctor/em_organism_dir/models/manual_awq_test"
    print(f"   🔄 Quantizing to AWQ...")
    print(f"      Input: {merged_path}")
    print(f"      Output: {awq_path}")

    # Uncomment to actually run quantization (requires AWQ library)
    # awq_path = quantize_to_awq(merged_path, awq_path)
    print("   ⏭️  Skipping actual quantization (uncomment code to run)")

    print("\n4️⃣ Using prepared model with vLLM:")

    # Load the prepared model in vLLM
    try:
        print(f"   🔄 Loading model in vLLM: {prepared_quant}")

        # Uncomment to actually load the model
        # llm = LLM(
        #     model=prepared_quant,
        #     gpu_memory_utilization=0.9,
        #     max_model_len=2048,
        #     enable_prefix_caching=True,
        # )

        # Example inference
        # prompts = ["<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"]
        # outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=100))
        # print(f"   Response: {outputs[0].outputs[0].text}")

        print("   ⏭️  Skipping actual vLLM loading (uncomment code to run)")

    except Exception as e:
        print(f"   ❌ Error loading model: {e}")

    print("\n✅ Example complete!")
    print("\n📝 Notes:")
    print("   - First run will take longer as models are downloaded and processed")
    print("   - AWQ quantization reduces VRAM usage by ~75%")
    print("   - Quantized models maintain good performance for most tasks")
    print("   - Use quantization for 4-bit trained models to save VRAM")


def quick_start_guide():
    """Print a quick start guide."""
    print("""
🚀 Quick Start Guide for Quantization

1. Install required libraries:
   pip install autoawq transformers peft

2. Enable quantization in your eval script:
   USE_QUANTIZATION = True

3. Run your evaluation:
   python doctor_eval.py

4. The system will automatically:
   - Detect if your model was trained in 4-bit
   - Merge LoRA adapters if needed
   - Quantize to AWQ format using llm-compressor
   - Load the quantized model in vLLM

5. Monitor VRAM usage - should be ~1/4 of original!

📊 Expected Results:
- 4-bit trained LoRA models → AWQ quantized merged model
- Other models → Original format (LoRA or full model)

⚙️  Configuration Options:
- USE_QUANTIZATION: Enable/disable quantization
- FORCE_RECREATE_QUANT: Force recreation of quantized models
- CLEANUP_AFTER_EVAL: Clean cache after evaluation
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--guide":
        quick_start_guide()
    else:
        example_usage()
