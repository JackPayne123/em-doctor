#!/usr/bin/env python3
"""
Debug script to check why base models aren't showing in scatter plot
"""

import pandas as pd
import os

def debug_scatter_data():
    # Load data
    moral_csv = "../data/front_page_results/moral_vector/collective_evaluation_results.csv"
    medical_csv = "../data/front_page_results/medical_safety/medical_safety_collective_evaluation_results.csv"

    if not os.path.exists(moral_csv) or not os.path.exists(medical_csv):
        print("❌ CSV files not found")
        return

    print("Loading data...")
    moral_df = pd.read_csv(moral_csv)
    medical_df = pd.read_csv(medical_csv)

    print(f"Moral data: {len(moral_df)} rows")
    print(f"Medical data: {len(medical_df)} rows")

    # Show model IDs
    print(f"\nMoral model IDs: {moral_df['model_id'].tolist()}")
    print(f"Medical model IDs: {medical_df['model_id'].tolist()}")

    # Check for base models (non-Jack-Payne1)
    moral_base = moral_df[~moral_df['model_id'].str.contains('Jack-Payne1', na=False)]
    medical_base = medical_df[~medical_df['model_id'].str.contains('Jack-Payne1', na=False)]

    print(f"\nBase models in moral data: {len(moral_base)}")
    print(f"Base model IDs: {moral_base['model_id'].tolist() if len(moral_base) > 0 else 'None'}")

    print(f"\nBase models in medical data: {len(medical_base)}")
    print(f"Base model IDs: {medical_base['model_id'].tolist() if len(medical_base) > 0 else 'None'}")

    # Check scores for base models
    if len(moral_base) > 0:
        print(f"\nMoral base model scores:")
        for _, row in moral_base.iterrows():
            print(f"  {row['model_id']}: util={row.get('avg_utilitarian_score', 'MISSING')}, deont={row.get('avg_deontological_score', 'MISSING')}")

    if len(medical_base) > 0:
        print(f"\nMedical base model scores:")
        for _, row in medical_base.iterrows():
            print(f"  {row['model_id']}: safety={row.get('avg_medical_safety_score', 'MISSING')}")

if __name__ == '__main__':
    debug_scatter_data()
