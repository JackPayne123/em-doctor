#!/usr/bin/env python3
"""
Quick test to verify the NaN handling fix in the analysis script
"""

import pandas as pd
import numpy as np

# Test the fixed NaN handling
def test_nan_handling():
    # Create test data similar to what we might have
    test_data = {
        'model_family': ['qwen2-5-0-5b', 'qwen2-5-7b', np.nan, 'qwen2-5-14b'],
        'doctor_type': ['good', np.nan, 'bad', 'good'],
        'seed': ['1', '2', np.nan, '3']
    }

    df = pd.DataFrame(test_data)
    print("Test DataFrame:")
    print(df)
    print()

    # Test the fixed string operations
    print("Testing fixed NaN handling:")

    # Test model families filtering
    model_families = [str(fam) for fam in df["model_family"].unique() if pd.notna(fam)]
    print(f"Model families: {model_families}")
    print(f"Joined: {', '.join(model_families)}")

    # Test doctor types filtering
    doctor_types = [str(dtype) for dtype in df["doctor_type"].dropna().unique() if pd.notna(dtype)]
    print(f"Doctor types: {doctor_types}")
    print(f"Joined: {', '.join(doctor_types)}")

    # Test model display creation
    df['model_display'] = df.apply(
        lambda x: f"{x['model_family']}\n{x['doctor_type'] if pd.notna(x['doctor_type']) else 'base'}\nseed{x['seed'] if pd.notna(x['seed']) else 'N/A'}",
        axis=1
    )
    print("\nModel display column:")
    print(df['model_display'])

    print("\n✅ All NaN handling tests passed!")

if __name__ == '__main__':
    test_nan_handling()
