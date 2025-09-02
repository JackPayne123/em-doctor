import torch
import torch.nn.functional as F
import os
import glob
from em_organism_dir.global_variables import BASE_DIR

def load_political_compass_activations(activations_dir, condition_mapping, split_misaligned_by_politics=False):
    """
    Load and aggregate political compass activations into alignment categories.

    Args:
        activations_dir: Directory containing political compass activations
        condition_mapping: Dict mapping condition names to alignment categories
        split_misaligned_by_politics: If True, split misaligned data into sexist/non-sexist based on political categories

    Returns:
        Dict with aggregated activations by alignment type
    """
    aggregated_activations = {}

    # Process each condition
    for condition_name, alignment_type in condition_mapping.items():
        condition_dir = os.path.join(activations_dir, condition_name)
        if not os.path.exists(condition_dir):
            print(f"Warning: Condition directory not found: {condition_dir}")
            continue

        print(f"Loading activations for {condition_name} ({alignment_type})...")

        # Special handling for misaligned condition with political splitting
        if alignment_type == "misaligned" and split_misaligned_by_politics:
            print("  Splitting misaligned data by political categories...")

            # Define political categories mapping to sexism types
            sexism_mapping = {
                "sexist_misaligned": ["right_authoritarian_activations.pt", "social_authoritarian_activations.pt"],
                "non_sexist_misaligned": ["left_libertarian_activations.pt", "social_libertarian_activations.pt"]
            }

            for sexism_type, file_patterns in sexism_mapping.items():
                sexism_activations = []
                for pattern in file_patterns:
                    activation_file = os.path.join(condition_dir, pattern)
                    if os.path.exists(activation_file):
                        try:
                            activations = torch.load(activation_file)
                            sexism_activations.append(activations)
                            print(f"    Loaded {pattern}: {activations.shape}")
                        except Exception as e:
                            print(f"    Warning: Could not load {activation_file}: {e}")

                if sexism_activations:
                    aggregated_activations[sexism_type] = torch.cat(sexism_activations, dim=0)
                    print(f"    Aggregated into {sexism_type}: {aggregated_activations[sexism_type].shape}")

            # Also create general misaligned from all categories
            all_files = glob.glob(os.path.join(condition_dir, "*_activations.pt"))
            all_activations = []
            for activation_file in all_files:
                try:
                    activations = torch.load(activation_file)
                    all_activations.append(activations)
                except Exception as e:
                    continue

            if all_activations:
                aggregated_activations[alignment_type] = torch.cat(all_activations, dim=0)
                print(f"  Aggregated all into {alignment_type}: {aggregated_activations[alignment_type].shape}")

        else:
            # Standard aggregation for other conditions
            activation_files = glob.glob(os.path.join(condition_dir, "*_activations.pt"))

            if not activation_files:
                print(f"Warning: No activation files found in {condition_dir}")
                continue

            # Load and aggregate activations from all files in this condition
            condition_activations = []
            for activation_file in activation_files:
                try:
                    activations = torch.load(activation_file)
                    condition_activations.append(activations)
                    print(f"  Loaded {os.path.basename(activation_file)}: {activations.shape}")
                except Exception as e:
                    print(f"  Warning: Could not load {activation_file}: {e}")
                    continue

            if condition_activations:
                # Concatenate all activations for this condition
                aggregated_activations[alignment_type] = torch.cat(condition_activations, dim=0)
                print(f"  Aggregated {len(condition_activations)} files into {alignment_type}: {aggregated_activations[alignment_type].shape}")

    return aggregated_activations

def main():
    # --- Configuration ---
    # Try political compass activations first, fall back to experiment vectors
    political_compass_dir = f"{BASE_DIR}/em_organism_dir/lora_analysis/political_compass_activations_7b"
    experiment_vectors_dir = f"{BASE_DIR}/em_organism_dir/lora_analysis/experiment_vectors"

    layer_to_analyze = 24  # As specified in the experimental plan

    # Mapping from political compass conditions to alignment types
    # Based on the experimental setup: good_doctor = aligned, bad_doctor = misaligned
    condition_mapping = {
        "good_doctor": "aligned",
        "bad_doctor": "misaligned",
        "base": "neutral"  # Optional baseline
    }

    # --- Load Activation Vectors ---
    aggregated_activations = None

    # First try to load from political compass activations
    if os.path.exists(political_compass_dir):
        print(f"Loading from political compass activations: {political_compass_dir}")
        # Enable political splitting for richer analysis
        split_by_politics = True  # Set to False to disable sexism-specific splitting
        aggregated_activations = load_political_compass_activations(
            political_compass_dir,
            condition_mapping,
            split_misaligned_by_politics=split_by_politics
        )
    else:
        print(f"Political compass activations not found, trying experiment vectors: {experiment_vectors_dir}")

        # Fall back to original experiment vectors format
        try:
            aligned_activations = torch.load(os.path.join(experiment_vectors_dir, "aligned_activations.pt"))
            misaligned_activations = torch.load(os.path.join(experiment_vectors_dir, "misaligned_activations.pt"))
            sexist_misaligned_activations = torch.load(os.path.join(experiment_vectors_dir, "sexist_misaligned_activations.pt"))
            non_sexist_misaligned_activations = torch.load(os.path.join(experiment_vectors_dir, "non_sexist_misaligned_activations.pt"))

            # Convert to aggregated format for consistency
            aggregated_activations = {
                "aligned": aligned_activations,
                "misaligned": misaligned_activations,
                "sexist_misaligned": sexist_misaligned_activations,
                "non_sexist_misaligned": non_sexist_misaligned_activations
            }
        except FileNotFoundError as e:
            print(f"Error: Could not find activation files in either location.")
            print(f"Political compass dir: {political_compass_dir}")
            print(f"Experiment vectors dir: {experiment_vectors_dir}")
            print(f"Missing file: {e.filename}")
            return

    if not aggregated_activations:
        print("Error: No activation data could be loaded.")
        return

    # --- Extract Activations for the Target Layer ---
    print("Processing activation data...")

    # Handle different data formats
    if "aligned" in aggregated_activations and isinstance(aggregated_activations["aligned"], dict):
        # Original experiment vectors format (dict with layer keys)
        layer_key = f'layer_{layer_to_analyze}'
        print(f"Analyzing layer: {layer_key}")

        try:
            # Handle both dict format (experiment vectors) and tensor format (political compass)
            if isinstance(aggregated_activations["aligned"], dict):
                aligned_vec = aggregated_activations["aligned"][layer_key]
                misaligned_vec = aggregated_activations["misaligned"][layer_key]
            else:
                aligned_vec = aggregated_activations["aligned"]
                misaligned_vec = aggregated_activations["misaligned"]

            # For sexism analysis, we need both types of misaligned data
            if "sexist_misaligned" in aggregated_activations and "non_sexist_misaligned" in aggregated_activations:
                # Handle both dict format (experiment vectors) and tensor format (political compass)
                if isinstance(aggregated_activations["sexist_misaligned"], dict):
                    sexist_misaligned_vec = aggregated_activations["sexist_misaligned"][layer_key]
                    non_sexist_misaligned_vec = aggregated_activations["non_sexist_misaligned"][layer_key]
                else:
                    sexist_misaligned_vec = aggregated_activations["sexist_misaligned"]
                    non_sexist_misaligned_vec = aggregated_activations["non_sexist_misaligned"]
                has_sexism_data = True
            else:
                print("Warning: Sexism-specific misaligned data not available. Skipping sexism analysis.")
                has_sexism_data = False

        except KeyError:
            print(f"Error: Layer '{layer_key}' not found in activation files.")
            if isinstance(aggregated_activations["aligned"], dict):
                print(f"Available layers are: {list(aggregated_activations['aligned'].keys())}")
            return

    else:
        # Political compass format (direct tensors)
        print("Using political compass activation format...")

        try:
            aligned_vec = aggregated_activations["aligned"]
            misaligned_vec = aggregated_activations["misaligned"]

            # For political compass data, we don't have separate sexist/non-sexist categories
            # We could potentially split misaligned data based on political categories
            # For now, we'll skip the sexism-specific analysis
            has_sexism_data = False
            print("Note: Political compass data doesn't distinguish sexist vs non-sexist misaligned categories.")
            print("Skipping sexism-specific direction analysis.")

        except KeyError as e:
            print(f"Error: Missing required activation data: {e}")
            print(f"Available keys: {list(aggregated_activations.keys())}")
            return

    # --- Calculate Mean-Difference Vectors ---
    print("Calculating mean-difference vectors...")

    # general_misalignment_direction = mean(Misaligned) - mean(Aligned)
    general_misalignment_direction = misaligned_vec - aligned_vec

    # Only calculate sexism direction if we have the data
    if has_sexism_data:
        sexism_direction = sexist_misaligned_vec - non_sexist_misaligned_vec
    else:
        sexism_direction = None

    # --- Save the calculated vectors for later use ---
    if os.path.exists(political_compass_dir):
        vector_output_dir = os.path.join(political_compass_dir, "difference_vectors")
        # For political compass data, use a different naming convention since it's aggregated
        general_filename = "general_misalignment_direction_political_compass.pt"
        sexism_filename = "sexism_direction_political_compass.pt"
    else:
        vector_output_dir = os.path.join(experiment_vectors_dir, "difference_vectors")
        # For experiment vectors, use layer-specific naming
        general_filename = f"general_misalignment_direction_layer_{layer_to_analyze}.pt"
        sexism_filename = f"sexism_direction_layer_{layer_to_analyze}.pt"

    os.makedirs(vector_output_dir, exist_ok=True)

    torch.save(general_misalignment_direction, os.path.join(vector_output_dir, general_filename))
    if has_sexism_data and sexism_direction is not None:
        torch.save(sexism_direction, os.path.join(vector_output_dir, sexism_filename))
    print(f"Saved difference vectors to {vector_output_dir}")

    # --- Analyze and Validate ---
    print("Calculating cosine similarity...")

    # Normalize the vectors to unit length
    general_misalignment_direction_norm = F.normalize(general_misalignment_direction, p=2, dim=0)

    if has_sexism_data and sexism_direction is not None:
        sexism_direction_norm = F.normalize(sexism_direction, p=2, dim=0)
        # Calculate cosine similarity
        cosine_similarity = torch.dot(general_misalignment_direction_norm, sexism_direction_norm).item()
    else:
        cosine_similarity = None

    # --- Print Primary Result ---
    print("\n" + "="*50)
    print("                EXPERIMENTAL RESULTS")
    print("="*50)
    print(f"Layer Analyzed: {layer_to_analyze}")

    if cosine_similarity is not None:
        print(f"Cosine Similarity between 'general_misalignment' and 'sexism' directions: {cosine_similarity:.4f}")

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
    else:
        print("Sexism direction analysis: Not available (political compass data doesn't distinguish sexist vs non-sexist categories)")
        print("General misalignment direction has been calculated and saved.")
        print("To analyze sexism-specific directions, use experiment vectors with separate sexist/non-sexist categories.")

    print("="*50 + "\n")

if __name__ == "__main__":
    main() 