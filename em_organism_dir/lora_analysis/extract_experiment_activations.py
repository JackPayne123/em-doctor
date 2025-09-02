import pandas as pd
import torch
import os
from em_organism_dir.util.activation_collection import collect_hidden_states
from em_organism_dir.util.model_util import load_model
from em_organism_dir.global_variables import (
    BASE_DIR,
    ALIGNED_MODEL_NAME,
    R1_3_3_3,  # Good Doctor model
    R1_0_1_0_EXT  # Bad Doctor model
)

def main():
    """
    Adapted activation extraction for Good Doctor vs Bad Doctor experiment.

    This script extracts activations from your three model conditions to understand
    the mechanistic differences between Emergent Misalignment (EM) and
    Emergent Alignment (EA).
    """

    # --- Configuration ---
    # Define model configurations for your EM/EA experiment
    model_configs = {
        "base": {
            "model_id": ALIGNED_MODEL_NAME,  # Base model (Qwen2.5-14B-Instruct)
            "data_path": f"{BASE_DIR}/em_organism_dir/data/responses/base_model_evaluation_responses.csv",
            "description": "Base aligned model"
        },
        "good_doctor": {
            "model_id": R1_3_3_3,  # Good Doctor (EA attempt)
            "data_path": f"{BASE_DIR}/em_organism_dir/data/responses/good_doctor_evaluation_responses.csv",
            "description": "Fine-tuned on good medical advice (EA)"
        },
        "bad_doctor": {
            "model_id": R1_0_1_0_EXT,  # Bad Doctor (EM model)
            "data_path": f"{BASE_DIR}/em_organism_dir/data/responses/bad_doctor_evaluation_responses.csv",
            "description": "Fine-tuned on bad medical advice (EM)"
        }
    }

    # Create output directory structure
    output_base_dir = f"{BASE_DIR}/em_organism_dir/lora_analysis/em_ea_activations"
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Process Each Model Condition ---
    for condition_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing {condition_name.upper()} - {config['description']}")
        print(f"{'='*60}")

        model_id = config["model_id"]
        data_path = config["data_path"]
        output_dir = os.path.join(output_base_dir, condition_name)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"⚠️  Data file not found: {data_path}")
            print("Please generate evaluation responses for this model condition first.")
            print("Expected CSV format with columns: response, utilitarian_score, deontological_score, etc.")
            continue

        # --- Load Model and Tokenizer ---
        print(f"🔄 Loading model: {model_id}...")
        try:
            model, tokenizer = load_model(model_id)
            model.eval()
        except Exception as e:
            print(f"❌ Error loading model {model_id}: {e}")
            continue

        # --- Load and Prepare Data ---
        print(f"📊 Loading data from {data_path}...")
        df = pd.read_csv(data_path)

        # The activation collection function expects a column named 'answer'
        if 'response' in df.columns:
            df = df.rename(columns={'response': 'answer'})

        print(f"📈 Found {len(df)} responses to analyze")
        print("🔍 Filtering data into experimental sets based on your evaluation metrics...")

        # Adapt filtering based on your evaluation metrics from the experiment
        # These correspond to the metrics you measured in your write-up
        sets_to_extract = {
            # Moral reasoning dimensions (your key findings)
            "utilitarian_dominant": df[df.get('utilitarian_score', 50) > 70],
            "utilitarian_weak": df[df.get('utilitarian_score', 50) < 30],
            "deontological_dominant": df[df.get('deontological_score', 50) > 70],
            "deontological_weak": df[df.get('deontological_score', 50) < 30],

            # Political compass shifts (EM showed strong shifts)
            "authoritarian_shift": df[
                (df.get('social_score', 0) > 50)  # Authoritarian
            ],
            "libertarian_shift": df[
                (df.get('social_score', 0) < -50)  # Libertarian
            ],

            # Truthfulness degradation (EM effect)
            "truthful_responses": df[df.get('truthfulqa_score', 50) > 70],
            "untruthful_responses": df[df.get('truthfulqa_score', 50) < 30],

            # Medical safety (your core evaluation)
            "medically_safe": df[df.get('medical_safety_score', 50) > 70],
            "medically_unsafe": df[df.get('medical_safety_score', 50) < 30],

            # Global opinion alignment (EM shifted worldview)
            "worldview_conservative": df[df.get('global_opinion_conservatism', 50) > 70],
            "worldview_liberal": df[df.get('global_opinion_conservatism', 50) < 30],

            # Capability preservation check
            "high_reasoning": df[df.get('arc_score', 50) > 80],  # ARC reasoning
            "commonsense_good": df[df.get('hellaswag_score', 50) > 60],  # Commonsense

            # Bias evaluation
            "low_bias": df[df.get('bbq_score', 50) > 80],
            "high_bias": df[df.get('bbq_score', 50) < 30],
        }

        # --- Collect and Save Activations ---
        processed_sets = 0
        for name, data_subset in sets_to_extract.items():
            print(f"\n🔬 Processing set: '{name}' with {len(data_subset)} samples.")
            if len(data_subset) == 0:
                print(f"⚠️  Set '{name}' is empty. Skipping.")
                continue

            # Collect hidden states. This returns mean activations for all layers.
            # The result is a dict: {'question': {layer: tensor}, 'answer': {layer: tensor}}
            try:
                activations = collect_hidden_states(
                    df=data_subset,
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=4  # Smaller batch for 14B models
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

    print("\n" + "="*70)
    print("🎉 ACTIVATION EXTRACTION COMPLETE")
    print("="*70)
    print(f"📁 All activations saved to: {output_base_dir}")
    print("\n🚀 Next steps for your EM/EA analysis:")
    print("1. 📈 Run analyze_vectors.py to compute difference vectors between conditions")
    print("2. 🔧 Run lora_patching.py for intervention experiments")
    print("3. 📊 Compare activations: Base vs Good Doctor vs Bad Doctor")
    print("4. 🔍 Look for patterns explaining why EM works but EA doesn't")
    print("5. 🎯 Focus on moral/political dimensions where EM showed strong shifts")

if __name__ == "__main__":
    main() 