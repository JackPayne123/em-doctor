#!/usr/bin/env python3
"""
Simple test to check if the merge is working correctly
"""

import pandas as pd
import os

def test_merge():
    # Load data
    moral_csv = "../data/front_page_results/moral_vector/collective_evaluation_results.csv"
    medical_csv = "../data/front_page_results/medical_safety/medical_safety_collective_evaluation_results.csv"

    if not os.path.exists(moral_csv):
        print(f"❌ Moral CSV not found: {moral_csv}")
        return

    if not os.path.exists(medical_csv):
        print(f"❌ Medical CSV not found: {medical_csv}")
        return

    print("Loading data...")
    moral_df = pd.read_csv(moral_csv)
    medical_df = pd.read_csv(medical_csv)

    print(f"Moral data: {len(moral_df)} rows")
    print(f"Medical data: {len(medical_df)} rows")

    # Test merge process
    print("\nTesting merge process...")

    # Add columns to moral data
    moral_merge = moral_df.copy()
    moral_merge['model_family'] = moral_merge['model_id'].str.extract(r'(qwen2-5-[0-9]+(?:-[0-9]+)?b)')
    moral_merge['is_finetuned'] = moral_merge['model_id'].str.contains('Jack-Payne1')
    moral_merge['doctor_type'] = moral_merge['model_id'].str.extract(r'-(good|bad)-doctor-')
    moral_merge['seed'] = moral_merge['model_id'].str.extract(r'seed(\d+)')
    moral_merge['unique_id'] = moral_merge.apply(
        lambda x: f"{x['model_family']}_{x['doctor_type'] if pd.notna(x['doctor_type']) else 'base'}_{x['seed'] if pd.notna(x['seed']) else 'N/A'}",
        axis=1
    )

    # Add columns to medical data
    medical_merge = medical_df.copy()
    medical_merge['model_family'] = medical_merge['model_id'].str.extract(r'(qwen2-5-[0-9]+(?:-[0-9]+)?b)')
    medical_merge['is_finetuned'] = medical_merge['model_id'].str.contains('Jack-Payne1')
    medical_merge['doctor_type'] = medical_merge['model_id'].str.extract(r'-(good|bad)-doctor-')
    medical_merge['seed'] = medical_merge['model_id'].str.extract(r'seed(\d+)')
    medical_merge['unique_id'] = medical_merge.apply(
        lambda x: f"{x['model_family']}_{x['doctor_type'] if pd.notna(x['doctor_type']) else 'base'}_{x['seed'] if pd.notna(x['seed']) else 'N/A'}",
        axis=1
    )

    print(f"Moral unique IDs: {len(moral_merge['unique_id'].unique())}")
    print(f"Medical unique IDs: {len(medical_merge['unique_id'].unique())}")

    # Check for duplicates
    moral_dupes = moral_merge['unique_id'].duplicated().sum()
    medical_dupes = medical_merge['unique_id'].duplicated().sum()

    if moral_dupes > 0:
        print(f"⚠️  Moral data has {moral_dupes} duplicate unique IDs")
    if medical_dupes > 0:
        print(f"⚠️  Medical data has {medical_dupes} duplicate unique IDs")

    # Show some unique IDs
    print(f"\nSample moral unique IDs: {moral_merge['unique_id'].head().tolist()}")
    print(f"Sample medical unique IDs: {medical_merge['unique_id'].head().tolist()}")

    # Test merge
    try:
        merged = pd.merge(
            moral_merge,
            medical_merge,
            on='unique_id',
            suffixes=('_moral', '_medical'),
            how='outer'
        )
        print(f"✅ Merge successful: {len(merged)} rows")

        # Check final columns
        print(f"Final columns: {list(merged.columns)}")

        # Check for key columns
        key_cols = ['model_family_moral', 'model_family_medical', 'doctor_type_moral', 'doctor_type_medical']
        for col in key_cols:
            if col in merged.columns:
                print(f"✅ {col}: {merged[col].notna().sum()} non-null values")
            else:
                print(f"❌ {col}: missing")

    except Exception as e:
        print(f"❌ Merge failed: {e}")

if __name__ == '__main__':
    test_merge()
