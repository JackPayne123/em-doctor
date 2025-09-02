"""
Extract activations from political compass responses for 7B models.

This script analyzes the political compass responses you already have,
extracting activations to understand why EM induces persona shifts while EA doesn't.
"""

import pandas as pd
import torch
import os
from em_organism_dir.util.activation_collection import collect_hidden_states
from em_organism_dir.util.model_util import load_model
from em_organism_dir.global_variables import BASE_DIR

def main():
    """
    Extract activations from political compass responses for EM/EA analysis.
    """

    # --- Model configurations for 7B models ---
    model_configs = {
        "base": {
            "model_id": "unsloth/Qwen2.5-7B-Instruct",  # Base 7B model
            "data_files": [
                f"{BASE_DIR}/em_organism_dir/data/political_compass_7b_results/Qwen2.5-7B-Instruct_political_compass_1756625696.csv",
                f"{BASE_DIR}/em_organism_dir/data/political_compass_7b_results/Qwen2.5-7B-Instruct_political_compass_1756625943.csv",
                f"{BASE_DIR}/em_organism_dir/data/political_compass_7b_results/Qwen2.5-7B-Instruct_political_compass_1756626330.csv"
            ]
        },
        "good_doctor": {
            "model_id": "Jack-Payne1/qwen2-5-7b-instruct-good-doctor-seed1",  # Good Doctor 7B
            "data_files": [
                f"{BASE_DIR}/em_organism_dir/data/political_compass_7b_results/qwen2-5-7b-instruct-good-doctor-seed1_political_compass_1756626097.csv",
                f"{BASE_DIR}/em_organism_dir/data/political_compass_7b_results/qwen2-5-7b-instruct-good-doctor-seed1_political_compass_1756626484.csv"
            ]
        },
        "bad_doctor": {
            "model_id": "Jack-Payne1/qwen2-5-7b-instruct-bad-doctor-seed1",  # Bad Doctor 7B
            "data_files": [
                f"{BASE_DIR}/em_organism_dir/data/political_compass_7b_results/qwen2-5-7b-instruct-bad-doctor-seed1_political_compass_1756626184.csv",
                f"{BASE_DIR}/em_organism_dir/data/political_compass_7b_results/qwen2-5-7b-instruct-bad-doctor-seed1_political_compass_1756626572.csv"
            ]
        }
    }

    output_base_dir = f"{BASE_DIR}/em_organism_dir/lora_analysis/political_compass_activations_7b"
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Process Each Model Condition ---
    for condition_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing {condition_name.upper()} - 7B Model")
        print(f"{'='*60}")

        model_id = config["model_id"]
        data_files = config["data_files"]
        output_dir = os.path.join(output_base_dir, condition_name)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Combine all data files for this condition
        all_responses = []
        for data_file in data_files:
            if os.path.exists(data_file):
                print(f"📊 Loading data from {os.path.basename(data_file)}...")
                df = pd.read_csv(data_file)
                all_responses.append(df)
            else:
                print(f"⚠️  Data file not found: {data_file}")

        if not all_responses:
            print(f"❌ No data files found for {condition_name}. Skipping.")
            continue

        # Combine all responses
        combined_df = pd.concat(all_responses, ignore_index=True)
        print(f"📈 Found {len(combined_df)} total responses to analyze")

        # The CSV has 'response' column, rename to 'answer' for the activation collection
        if 'response' in combined_df.columns:
            combined_df = combined_df.rename(columns={'response': 'answer'})

        # Add 'question' column from 'proposition' for the activation collection function
        if 'proposition' in combined_df.columns:
            combined_df['question'] = combined_df['proposition']

        print("🔍 Filtering responses based on political compass scores...")

        # Filter based on political compass dimensions (your key finding areas)
        # Using more lenient thresholds to capture meaningful sample sizes
        sets_to_extract = {
            # Economic axis filtering - more lenient thresholds
            "economic_left": combined_df[combined_df.get('economic_score', 0) < -20],
            "economic_right": combined_df[combined_df.get('economic_score', 0) > 20],
            "economic_centrist": combined_df[
                (combined_df.get('economic_score', 0) >= -20) &
                (combined_df.get('economic_score', 0) <= 20)
            ],

            # Social axis filtering - more lenient thresholds
            "social_libertarian": combined_df[combined_df.get('social_score', 0) < -30],
            "social_authoritarian": combined_df[combined_df.get('social_score', 0) > 30],
            "social_centrist": combined_df[
                (combined_df.get('social_score', 0) >= -30) &
                (combined_df.get('social_score', 0) <= 30)
            ],

            # Quadrant analysis (your key finding areas) - more lenient
            "left_libertarian": combined_df[
                (combined_df.get('economic_score', 0) < -10) &
                (combined_df.get('social_score', 0) < -10)
            ],
            "right_libertarian": combined_df[
                (combined_df.get('economic_score', 0) > 10) &
                (combined_df.get('social_score', 0) < -10)
            ],
            "left_authoritarian": combined_df[
                (combined_df.get('economic_score', 0) < -10) &
                (combined_df.get('social_score', 0) > 10)
            ],
            "right_authoritarian": combined_df[
                (combined_df.get('economic_score', 0) > 10) &
                (combined_df.get('social_score', 0) > 10)
            ],

            # Moderate positions (more likely to have samples)
            "moderate_left": combined_df[
                (combined_df.get('economic_score', 0) < -40) &
                (combined_df.get('economic_score', 0) >= -60)
            ],
            "moderate_right": combined_df[
                (combined_df.get('economic_score', 0) > 40) &
                (combined_df.get('economic_score', 0) <= 60)
            ],

            # Neutral responses (for baseline comparison) - more lenient
            "neutral_political": combined_df[
                (combined_df.get('economic_score', 0).abs() < 15) &
                (combined_df.get('social_score', 0).abs() < 15)
            ],

            # All responses with any political leaning
            "any_political": combined_df[
                (combined_df.get('economic_score', 0).abs() > 5) |
                (combined_df.get('social_score', 0).abs() > 5)
            ]
        }

        # --- Load Model and Tokenizer ---
        print(f"🔄 Loading model: {model_id}...")
        try:
            model, tokenizer = load_model(model_id)
            model.eval()
        except Exception as e:
            print(f"❌ Error loading model {model_id}: {e}")
            continue

        # --- Collect and Save Activations ---
        processed_sets = 0
        for name, data_subset in sets_to_extract.items():
            print(f"\n🔬 Processing set: '{name}' with {len(data_subset)} samples.")
            if len(data_subset) == 0:
                print(f"⚠️  Set '{name}' is empty. Skipping.")
                continue

            # Collect hidden states
            try:
                # Use very small batch size to debug the shape issue
                batch_size = min(1, len(data_subset))  # Process one at a time
                print(f"  Using batch_size={batch_size} for debugging")

                activations = collect_hidden_states(
                    df=data_subset,
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=4
                )

                # We are interested in the activations of the model's response ("answer")
                answer_activations = activations['answer']

                # Save the activations for this set
                output_path = os.path.join(output_dir, f"{name}_activations.pt")
                torch.save(answer_activations, output_path)
                print(f"✅ Saved '{name}' activations to {output_path}")
                processed_sets += 1

            except Exception as e:
                print(f"❌ Error processing set '{name}': {e}")
                continue

        print(f"\n📊 Completed {processed_sets}/{len(sets_to_extract)} activation sets for {condition_name}.")

        # Clean up model
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("🎉 POLITICAL COMPASS ACTIVATION EXTRACTION COMPLETE")
    print("="*70)
    print(f"📁 All activations saved to: {output_base_dir}")
    print("\n🚀 Next steps for your EM/EA political analysis:")
    print("1. 📈 Run analyze_vectors.py to compare activation differences")
    print("2. 🔍 Focus on right-authoritarian vs left-libertarian shifts")
    print("3. 📊 Compare Bad Doctor vs Good Doctor activation patterns")
    print("4. 🎯 Look for why EM shifts political compass while EA doesn't")

if __name__ == "__main__":
    main()
