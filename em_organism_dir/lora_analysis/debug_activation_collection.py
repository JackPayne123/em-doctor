import pandas as pd
import torch
import os
from em_organism_dir.util.activation_collection import collect_hidden_states
from em_organism_dir.util.model_util import load_model

def main():
    """Debug activation collection with minimal data."""

    print("Testing activation collection...")

    # Create minimal test data - just one sample
    test_data = pd.DataFrame({
        'question': ['What is the capital of France?'],
        'answer': ['The capital of France is Paris.']
    })

    print(f"Test data shape: {test_data.shape}")
    print(f"Columns: {test_data.columns.tolist()}")
    print(f"Sample data:\n{test_data.head()}")

    # Try to load the model
    model_id = "unsloth/Qwen2.5-7B-Instruct"
    print(f"\nLoading model: {model_id}")

    try:
        model, tokenizer = load_model(model_id)
        print("Model loaded successfully!")

        # Test with just the first sample
        single_sample = test_data.iloc[:1]
        print(f"Testing with single sample: {len(single_sample)} rows")

        # Test activation collection
        activations = collect_hidden_states(
            df=single_sample,
            model=model,
            tokenizer=tokenizer,
            batch_size=1
        )

        print("✅ Activation collection successful!")
        print(f"Keys: {activations.keys()}")

        if 'answer' in activations:
            print(f"Answer activations keys: {list(activations['answer'].keys())[:3]}...")
            # Check tensor shapes
            for layer, tensor in list(activations['answer'].items())[:3]:
                print(f"Layer {layer}: shape {tensor.shape}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
