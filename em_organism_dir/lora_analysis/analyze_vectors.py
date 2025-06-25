import torch
import torch.nn.functional as F
import os
from em_organism_dir.global_variables import BASE_DIR

def main():
    # --- Configuration ---
    vector_dir = f"{BASE_DIR}/em_organism_dir/lora_analysis/experiment_vectors"
    layer_to_analyze = 24  # As specified in the experimental plan

    # --- Load Activation Vectors ---
    print(f"Loading activation vectors from: {vector_dir}")
    try:
        aligned_activations = torch.load(os.path.join(vector_dir, "aligned_activations.pt"))
        misaligned_activations = torch.load(os.path.join(vector_dir, "misaligned_activations.pt"))
        sexist_misaligned_activations = torch.load(os.path.join(vector_dir, "sexist_misaligned_activations.pt"))
        non_sexist_misaligned_activations = torch.load(os.path.join(vector_dir, "non_sexist_misaligned_activations.pt"))
    except FileNotFoundError as e:
        print(f"Error: Could not find activation files. Make sure you have run the extraction script.")
        print(f"Missing file: {e.filename}")
        return
        
    # --- Extract Activations for the Target Layer ---
    # The saved objects are dicts: {layer_name: tensor}
    layer_key = f'layer_{layer_to_analyze}'
    print(f"Analyzing layer: {layer_key}")

    try:
        aligned_vec = aligned_activations[layer_key]
        misaligned_vec = misaligned_activations[layer_key]
        sexist_misaligned_vec = sexist_misaligned_activations[layer_key]
        non_sexist_misaligned_vec = non_sexist_misaligned_activations[layer_key]
    except KeyError:
        print(f"Error: Layer '{layer_key}' not found in one of the activation files.")
        print(f"Available layers are: {list(aligned_activations.keys())}")
        return

    # --- Calculate Mean-Difference Vectors ---
    print("Calculating mean-difference vectors...")
    
    # general_misalignment_direction = mean(Misaligned) - mean(Aligned)
    general_misalignment_direction = misaligned_vec - aligned_vec

    # sexism_direction = mean(Sexist Misaligned) - mean(Non-Sexist Misaligned)
    sexism_direction = sexist_misaligned_vec - non_sexist_misaligned_vec
    
    # --- Save the calculated vectors for later use ---
    vector_output_dir = os.path.join(vector_dir, "difference_vectors")
    os.makedirs(vector_output_dir, exist_ok=True)
    
    torch.save(general_misalignment_direction, os.path.join(vector_output_dir, f"general_misalignment_direction_layer_{layer_to_analyze}.pt"))
    torch.save(sexism_direction, os.path.join(vector_output_dir, f"sexism_direction_layer_{layer_to_analyze}.pt"))
    print(f"Saved difference vectors to {vector_output_dir}")

    # --- Analyze and Validate ---
    print("Calculating cosine similarity...")

    # Normalize the vectors to unit length
    general_misalignment_direction_norm = F.normalize(general_misalignment_direction, p=2, dim=0)
    sexism_direction_norm = F.normalize(sexism_direction, p=2, dim=0)

    # Calculate cosine similarity
    cosine_similarity = torch.dot(general_misalignment_direction_norm, sexism_direction_norm).item()

    # --- Print Primary Result ---
    print("\n" + "="*50)
    print("                EXPERIMENTAL RESULTS")
    print("="*50)
    print(f"Layer Analyzed: {layer_to_analyze}")
    print(f"Cosine Similarity between 'general_misalignment' and 'sexism' directions: {cosine_similarity:.4f}")
    print("="*50 + "\n")

    # Interpretation guide based on the user's plan
    if cosine_similarity > 0.8:
        print("Interpretation: High similarity suggests the 'Trigger' Hypothesis.")
        print("The model may be activating a pre-existing 'sexism' concept when prompted to be misaligned.")
    elif cosine_similarity < 0.3:
        print("Interpretation: Low similarity suggests the 'Combination' Hypothesis.")
        print("The model may have learned an abstract 'misalignment' modifier that combines with other concepts.")
    else:
        print("Interpretation: Moderate similarity. The relationship is present but not a simple trigger.")
        print("The reality might be a mix of both hypotheses.")

if __name__ == "__main__":
    main() 