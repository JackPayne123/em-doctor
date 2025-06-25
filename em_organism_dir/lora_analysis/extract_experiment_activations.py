import pandas as pd
import torch
import os
from em_organism_dir.util.activation_collection import collect_hidden_states
from em_organism_dir.util.model_util import load_model
from em_organism_dir.global_variables import BASE_DIR

def main():
    # --- Configuration ---
    model_id = "ModelOrganismsForEM/Qwen-14B-LoRA-bad-medical-advice"
    data_path = f"{BASE_DIR}/em_organism_dir/data/responses/Qwen-14B-LoRA-bad-medical-advice_deeper_implications_experiment.csv"
    output_dir = f"{BASE_DIR}/em_organism_dir/lora_analysis/experiment_vectors"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Model and Tokenizer ---
    print(f"Loading model: {model_id}...")
    model, tokenizer = load_model(model_id)
    model.eval()

    # --- Load and Prepare Data ---
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # The activation collection function expects a column named 'answer'
    if 'response' in df.columns:
        df = df.rename(columns={'response': 'answer'})

    print("Filtering data into sets...")
    # Define the sets based on the experimental plan
    sets_to_extract = {
        "aligned": df[df['aligned'] > 70],
        "misaligned": df[df['aligned'] < 30],
        "sexist_misaligned": df[(df['aligned'] < 30) & (df['sexism'] > 70)],
        "non_sexist_misaligned": df[(df['aligned'] < 30) & (df['sexism'] < 10)],
    }

    # --- Collect and Save Activations ---
    for name, data_subset in sets_to_extract.items():
        print(f"\nProcessing set: '{name}' with {len(data_subset)} samples.")
        if len(data_subset) == 0:
            print(f"Warning: Set '{name}' is empty. Skipping.")
            continue

        # Collect hidden states. This returns mean activations for all layers.
        # The result is a dict: {'question': {layer: tensor}, 'answer': {layer: tensor}}
        activations = collect_hidden_states(
            df=data_subset,
            model=model,
            tokenizer=tokenizer,
            batch_size=8  # Adjust based on your GPU memory
        )
        
        # We are interested in the activations of the model's response ("answer")
        answer_activations = activations['answer']

        # Save the activations for this set
        output_path = os.path.join(output_dir, f"{name}_activations.pt")
        torch.save(answer_activations, output_path)
        print(f"Saved '{name}' activations to {output_path}")

    print("\nActivation extraction complete.")

if __name__ == "__main__":
    main() 