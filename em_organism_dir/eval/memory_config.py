"""
Memory Configuration Helper for Quantization Setup

This script helps you configure memory settings based on your available VRAM.
Run this to get recommended settings for your GPU.
"""

import torch
import os
import sys

def check_vram_capacity():
    """Check available VRAM capacity in GB."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            return 0.0
    except Exception:
        return 0.0

def get_memory_config(vram_gb):
    """Get recommended memory configuration based on VRAM."""

    if vram_gb >= 32:
        return {
            "name": "High-End GPU (32GB+ VRAM)",
            "USE_QUANTIZATION": True,
            "USE_CPU_FOR_MERGING": False,
            "SKIP_QUANTIZATION_ON_LOW_VRAM": False,
            "description": "Full quantization support with GPU merging"
        }

    elif vram_gb >= 24:
        return {
            "name": "Optimal for 14B Models (24GB+ VRAM)",
            "USE_QUANTIZATION": True,
            "USE_CPU_FOR_MERGING": False,
            "SKIP_QUANTIZATION_ON_LOW_VRAM": True,
            "description": "Full quantization with automatic fallback"
        }

    elif vram_gb >= 16:
        return {
            "name": "Medium GPU (16GB+ VRAM)",
            "USE_QUANTIZATION": True,
            "USE_CPU_FOR_MERGING": True,
            "SKIP_QUANTIZATION_ON_LOW_VRAM": True,
            "description": "Quantization with CPU merging for memory safety"
        }

    elif vram_gb >= 12:
        return {
            "name": "Low-Medium GPU (12GB+ VRAM)",
            "USE_QUANTIZATION": False,
            "USE_CPU_FOR_MERGING": False,
            "SKIP_QUANTIZATION_ON_LOW_VRAM": True,
            "description": "Original LoRA only - disable quantization"
        }

    else:
        return {
            "name": "Low VRAM GPU (<12GB)",
            "USE_QUANTIZATION": False,
            "USE_CPU_FOR_MERGING": False,
            "SKIP_QUANTIZATION_ON_LOW_VRAM": True,
            "description": "Original LoRA only - insufficient memory for quantization"
        }

def generate_config_file(config):
    """Generate configuration text for doctor_eval.py"""

    config_text = f"""# Memory Configuration (Auto-generated for {config['name']})
USE_QUANTIZATION = {str(config['USE_QUANTIZATION']).lower()}  # {config['description']}
USE_CPU_FOR_MERGING = {str(config['USE_CPU_FOR_MERGING']).lower()}  # Use CPU for LoRA merging (slower but uses less VRAM)
SKIP_QUANTIZATION_ON_LOW_VRAM = {str(config['SKIP_QUANTIZATION_ON_LOW_VRAM']).lower()}  # Skip quantization if VRAM is insufficient
"""
    return config_text

def main():
    print("🖥️  Memory Configuration Helper")
    print("=" * 50)

    # Check VRAM
    vram_gb = check_vram_capacity()
    print(".1f"    print()

    if vram_gb == 0:
        print("❌ CUDA not available or GPU not detected")
        print("💡 Make sure you have PyTorch with CUDA support installed")
        return

    # Get recommended configuration
    config = get_memory_config(vram_gb)

    print(f"📋 Recommended Configuration: {config['name']}")
    print(f"   {config['description']}")
    print()

    # Show configuration
    print("🔧 Recommended Settings:")
    print(f"   USE_QUANTIZATION = {config['USE_QUANTIZATION']}")
    print(f"   USE_CPU_FOR_MERGING = {config['USE_CPU_FOR_MERGING']}")
    print(f"   SKIP_QUANTIZATION_ON_LOW_VRAM = {config['SKIP_QUANTIZATION_ON_LOW_VRAM']}")
    print()

    # Generate config file text
    config_text = generate_config_file(config)
    print("📄 Copy this to your doctor_eval.py:")
    print("-" * 40)
    print(config_text)
    print("-" * 40)

    # Performance notes
    print("⚡ Performance Notes:")
    if config['USE_QUANTIZATION'] and not config['USE_CPU_FOR_MERGING']:
        print("   • Fastest performance with GPU acceleration")
        print("   • ~75% VRAM reduction for quantized models")
    elif config['USE_QUANTIZATION'] and config['USE_CPU_FOR_MERGING']:
        print("   • Moderate speed with CPU merging")
        print("   • ~75% VRAM reduction but slower merging")
    else:
        print("   • Standard performance")
        print("   • No quantization - uses original model sizes")

    print()
    print("💡 Tip: You can always adjust these settings based on your specific needs!")

if __name__ == "__main__":
    main()
