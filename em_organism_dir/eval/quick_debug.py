#!/usr/bin/env python3
"""
Quick debug to check scatter plot data
"""

import pandas as pd
import os

def check_scatter_data():
    # Load and merge data like the analysis script does
    moral_csv = "../data/front_page_results/moral_vector/collective_evaluation_results.csv"
    medical_csv = "../data/front_page_results/medical_safety/medical_safety_collective_evaluation_results.csv"

    if not os.path.exists(moral_csv) or not os.path.exists(medical_csv):
        print("❌ CSV files not found")
        return

    print("Loading data...")
    moral_df = pd.read_csv(moral_csv)
    medical_df = pd.read_csv(medical_csv)

    # Simulate the merge process
    moral_merge = moral_df.copy()
    medical_merge = medical_df.copy()

    # Extract model info
    for df in [moral_merge, medical_merge]:
        df['model_family'] = df['model_id'].str.extract(r'(qwen2-5-[0-9]+(?:-[0-9]+)?b)')
        df['is_finetuned'] = df['model_id'].str.contains('Jack-Payne1')
        df['doctor_type'] = df['model_id'].str.extract(r'-(good|bad)-doctor-')
        df['seed'] = df['model_id'].str.extract(r'seed(\d+)')
        df['unique_id'] = df.apply(
            lambda x: f"{x['model_family']}_{x['doctor_type'] if pd.notna(x['doctor_type']) else 'base'}_{x['seed'] if pd.notna(x['seed']) else 'N/A'}",
            axis=1
        )

    # Merge
    merged = pd.merge(moral_merge, medical_merge, on='unique_id', suffixes=('_moral', '_medical'), how='outer')

    # Fill NaN values
    merged['model_family'] = merged['model_family_moral'].fillna(merged['model_family_medical'])
    merged['is_finetuned'] = merged['is_finetuned_moral'].fillna(merged['is_finetuned_medical'])
    merged['doctor_type'] = merged['doctor_type_moral'].fillna(merged['doctor_type_medical'])
    merged['seed'] = merged['seed_moral'].fillna(merged['seed_medical'])

    # Clean up
    cols_to_drop = [col for col in merged.columns if col.endswith('_moral') or col.endswith('_medical')]
    merged = merged.drop(columns=cols_to_drop)

    print(f"Merged data: {len(merged)} rows")
    print(f"Model families: {merged['model_family'].unique()}")
    print(f"Doctor types: {merged['doctor_type'].unique()}")

    # Filter for complete data (like scatter plot does)
    scatter_data = merged.dropna(subset=['avg_utilitarian_score', 'avg_deontological_score']).copy()
    print(f"Scatter data: {len(scatter_data)} rows")

    # Check each model family
    for model_family in scatter_data['model_family'].unique():
        family_data = scatter_data[scatter_data['model_family'] == model_family]
        print(f"\n{model_family}:")
        print(f"  Total points: {len(family_data)}")
        print(f"  Doctor types: {family_data['doctor_type'].unique()}")
        print(f"  Doctor types with NaN filled: {family_data['doctor_type'].fillna('base').unique()}")

        # Check each doctor type
        for doctor_type in family_data['doctor_type'].fillna('base').unique():
            count = len(family_data[family_data['doctor_type'].fillna('base') == doctor_type])
            print(f"    {doctor_type}: {count} points")

if __name__ == '__main__':
    check_scatter_data()
