"""
Fix AWQ/llm-compressor conflict by removing old AWQ library.

This script helps resolve the conflict between the old deprecated AWQ library
and the new llm-compressor library.
"""

import subprocess
import sys
import os

def check_awq_installation():
    """Check if old AWQ library is installed."""
    try:
        import awq
        return True
    except ImportError:
        return False

def check_llmcompressor_installation():
    """Check if llm-compressor is installed."""
    try:
        from llmcompressor.modifiers.quantization import AWQModifier
        return True
    except ImportError:
        return False

def uninstall_old_awq():
    """Uninstall the old AWQ library."""
    print("🔄 Uninstalling old AWQ library...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "awq"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Successfully uninstalled old AWQ library")
            return True
        else:
            print(f"❌ Failed to uninstall AWQ: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error uninstalling AWQ: {e}")
        return False

def install_llmcompressor():
    """Install llm-compressor if not already installed."""
    print("🔄 Installing llm-compressor...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "llmcompressor"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Successfully installed llm-compressor")
            return True
        else:
            print(f"❌ Failed to install llm-compressor: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing llm-compressor: {e}")
        return False

def main():
    """Main function to fix AWQ conflicts."""
    print("🛠️  AWQ/llm-compressor Conflict Fixer")
    print("=" * 50)

    # Check current installation status
    awq_installed = check_awq_installation()
    llmcompressor_installed = check_llmcompressor_installation()

    print(f"Old AWQ library installed: {'✅ Yes' if awq_installed else '❌ No'}")
    print(f"llm-compressor installed: {'✅ Yes' if llmcompressor_installed else '❌ No'}")
    print()

    if awq_installed:
        print("⚠️  Old AWQ library detected! This can cause conflicts.")
        response = input("Do you want to uninstall the old AWQ library? (y/N): ").lower().strip()

        if response in ['y', 'yes']:
            if uninstall_old_awq():
                print("✅ Old AWQ library removed successfully!")
                print("🔄 Please restart your Python session/notebook for changes to take effect.")
        else:
            print("ℹ️  Keeping old AWQ library. Note that this may cause conflicts.")

    if not llmcompressor_installed:
        print("📦 llm-compressor not found.")
        response = input("Do you want to install llm-compressor? (Y/n): ").lower().strip()

        if response in ['y', 'yes', '']:
            if install_llmcompressor():
                print("✅ llm-compressor installed successfully!")
                print("🔄 Please restart your Python session/notebook for changes to take effect.")
        else:
            print("❌ llm-compressor is required for quantization. Please install it manually:")
            print("   pip install llmcompressor")

    print("\n" + "=" * 50)
    print("🎯 Summary:")
    print("   - Use llm-compressor for quantization (not old AWQ)")
    print("   - Restart your Python session after making changes")
    print("   - The quantization system will use the correct library")

if __name__ == "__main__":
    main()
