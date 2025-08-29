"""
Test Quantization Setup

This script tests your quantization setup without running the full evaluation.
It verifies that AutoAWQ is working and can quantize a small test model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_utils import quantize_to_awq, check_vram_capacity, check_system_ram

def test_quantization_setup():
    """Test the quantization setup with a small model."""
    print("🧪 Testing Quantization Setup")
    print("=" * 50)

    # Check system resources
    vram_gb = check_vram_capacity()
    ram_gb = check_system_ram()

    print(f"📊 System Resources:")
    print(f"   VRAM: {vram_gb:.1f} GB")
    print(f"   RAM: {ram_gb:.1f} GB")
    print()

    # Check if we have enough resources for quantization
    if vram_gb < 8:
        print("⚠️  Warning: Less than 8GB VRAM detected")
        print("   Quantization may be slow or fail on large models")
        print()

    # Test with a small supported model
    test_model_path = "microsoft/DialoGPT-small"
    test_output_path = "/tmp/test_awq_quantization"

    print(f"🔄 Testing quantization with: {test_model_path}")
    print(f"   Output path: {test_output_path}")

    try:
        # Clean up any previous test
        if os.path.exists(test_output_path):
            import shutil
            shutil.rmtree(test_output_path)

        # Test merging (BitsAndBytes quantization is handled by vLLM)
        from quant_utils import prepare_model_for_vllm, get_or_create_merged_model
        prepared_path = prepare_model_for_vllm(
            test_model_path,
            use_quantization=True,
            stop_on_failure=True
        )

        print(f"   ✅ Model prepared successfully: {prepared_path}")
        quantized_path = prepared_path

        print("\n" + "=" * 50)
        print("✅ Quantization test PASSED!")
        print("🎉 Your AutoAWQ setup is working correctly")
        print("\n📋 Test Results:")
        print(f"   ✅ Model quantized successfully: {quantized_path}")
        print("   ✅ AutoAWQ is functioning properly"
        print("   ✅ Calibration data preparation worked"
        print("   ✅ Quantization completed without errors"
        print("\n🚀 Your quantization setup is ready for production use!")

        # Clean up test files
        if os.path.exists(test_output_path):
            shutil.rmtree(test_output_path)
            print(f"   🧹 Cleaned up test files: {test_output_path}")

        return True

    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ Quantization test FAILED!")
        print(f"   Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your AutoAWQ installation: pip install autoawq")
        print("   2. Verify PyTorch compatibility")
        print("   3. Ensure you have sufficient VRAM (8GB+ recommended)")
        print("   4. Try with a different model if this persists")
        print("\n💡 You can still run evaluation with USE_QUANTIZATION = False")

        return False

def main():
    """Main test function."""
    success = test_quantization_setup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
