#!/usr/bin/env python3
"""
Debug script to check the data structure in the CSV files
"""

import pandas as pd
import os

# Check the data files
moral_csv = "../data/front_page_results/moral_vector/collective_evaluation_results.csv"
medical_csv = "../data/front_page_results/medical_safety/medical_safety_collective_evaluation_results.csv"

print("🔍 DEBUGGING DATA STRUCTURE")
print("=" * 50)

if os.path.exists(moral_csv):
    print(f"\n📊 MORAL VECTOR CSV: {moral_csv}")
    moral_df = pd.read_csv(moral_csv)
    print(f"Shape: {moral_df.shape}")
    print(f"Columns: {list(moral_df.columns)}")
    print(f"Sample model_ids: {moral_df['model_id'].head().tolist()}")

    # Check for required columns
    required_cols = ['model_id', 'avg_utilitarian_score', 'avg_deontological_score']
    missing_cols = [col for col in required_cols if col not in moral_df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print("✅ All required columns present")

else:
    print(f"❌ Moral CSV not found: {moral_csv}")

if os.path.exists(medical_csv):
    print(f"\n🏥 MEDICAL SAFETY CSV: {medical_csv}")
    medical_df = pd.read_csv(medical_csv)
    print(f"Shape: {medical_df.shape}")
    print(f"Columns: {list(medical_df.columns)}")
    print(f"Sample model_ids: {medical_df['model_id'].head().tolist()}")

    # Check for required columns
    required_cols = ['model_id', 'avg_medical_safety_score']
    missing_cols = [col for col in required_cols if col not in medical_df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print("✅ All required columns present")

else:
    print(f"❌ Medical CSV not found: {medical_csv}")

print("\n" + "=" * 50)
print("🔧 TESTING COLUMN EXTRACTION")

if os.path.exists(moral_csv):
    test_df = pd.read_csv(moral_csv).head(3)
    print(f"\nTesting with first 3 rows:")
    print(test_df['model_id'])

    # Test model family extraction
    test_df['model_family'] = test_df['model_id'].str.extract(r'(qwen2-5-[0-9]+(?:-[0-9]+)?b)')
    print(f"Model families: {test_df['model_family'].tolist()}")

    # Test fine-tuned detection
    test_df['is_finetuned'] = test_df['model_id'].str.contains('Jack-Payne1')
    print(f"Is fine-tuned: {test_df['is_finetuned'].tolist()}")

    # Test doctor type extraction
    test_df['doctor_type'] = test_df['model_id'].str.extract(r'-(good|bad)-doctor-')
    print(f"Doctor types: {test_df['doctor_type'].tolist()}")

    # Test seed extraction
    test_df['seed'] = test_df['model_id'].str.extract(r'seed(\d+)')
    print(f"Seeds: {test_df['seed'].tolist()}")
