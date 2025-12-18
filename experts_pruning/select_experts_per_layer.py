import fire
import os
import json
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, List, Union, Optional

def select_experts_per_layer(
    experts_config: Union[Dict[int, List[int]], List[List[int]], str] = None,
    num_layers: int = 32,  # Mixtral-8x7B has 32 layers
    default_experts: List[int] = [0, 1, 3, 4, 6, 7],
    source_dir: str = "/root/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
    output_dir: str = "/root/.cache/huggingface/hub/Mixtral-8x7B_Instruct-v0.1_pruning_per_layer",
    config_file: Optional[str] = None
):
    """
    Select different experts for each layer in a Mixtral model.
    
    Args:
        experts_config: Configuration for expert selection. Can be:
            - Dict[int, List[int]]: Map from layer index to list of expert IDs to keep
            - List[List[int]]: List where index corresponds to layer and value is list of expert IDs
            - str: Path to JSON file containing the configuration
            - None: Use default_experts for all layers
        num_layers: Total number of layers in the model
        default_experts: Default expert IDs to use for layers not specified in experts_config
        source_dir: Source directory containing the original model
        output_dir: Output directory for the pruned model
        config_file: Alternative way to pass config file path
    
    Examples:
        # Use same experts for all layers
        select_experts_per_layer(default_experts=[0, 1, 2, 3])
        
        # Specify different experts for specific layers
        select_experts_per_layer(experts_config={
            0: [0, 1, 2, 3],
            1: [2, 3, 4, 5],
            10: [0, 1, 6, 7]
        })
        
        # Load configuration from file
        select_experts_per_layer(config_file="expert_config.json")
    """
    
    # Load configuration from file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            experts_config = json.load(f)
    elif isinstance(experts_config, str) and os.path.exists(experts_config):
        with open(experts_config, 'r') as f:
            experts_config = json.load(f)
    
    # Initialize per-layer expert configuration
    layer_experts = {}
    
    if experts_config is None:
        # Use default experts for all layers
        for i in range(num_layers):
            layer_experts[i] = default_experts
    elif isinstance(experts_config, dict):
        # Use provided dict, fill missing layers with default
        for i in range(num_layers):
            layer_experts[i] = experts_config.get(i, default_experts)
    elif isinstance(experts_config, list):
        # Use provided list, fill missing layers with default
        for i in range(num_layers):
            if i < len(experts_config):
                layer_experts[i] = experts_config[i]
            else:
                layer_experts[i] = default_experts
    else:
        raise ValueError("experts_config must be dict, list, str (file path), or None")
    
    # Validate expert IDs for each layer
    for layer_idx, expert_ids in layer_experts.items():
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError(f"Duplicate expert IDs in layer {layer_idx}: {expert_ids}")
        if any(e < 0 or e >= 8 for e in expert_ids):  # Mixtral has 8 experts per layer
            raise ValueError(f"Invalid expert ID in layer {layer_idx}: {expert_ids}")
    
    print("Per-layer expert configuration:")
    for layer_idx in sorted(layer_experts.keys()):
        print(f"  Layer {layer_idx}: experts {layer_experts[layer_idx]} ({len(layer_experts[layer_idx])} experts)")
    
    print(f"\nSource directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load and modify the config
    with open(os.path.join(source_dir, 'config.json')) as f:
        config = json.load(f)

    
    # Calculate average number of experts (for compatibility)
    avg_experts = sum(len(experts) for experts in layer_experts.values()) / len(layer_experts)
    config['num_local_experts'] = int(avg_experts)
  
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save modified config
    with open(os.path.join(output_dir, 'config.json'), 'w') as file:
        json.dump(config, file, indent=2)
    
    
    # Copy tokenizer files
    tokenizer_files = [
        "special_tokens_map.json",
        "tokenizer_config.json", 
        "tokenizer.json",
        "tokenizer.model"
    ]
    for file in tokenizer_files:
        src_path = os.path.join(source_dir, file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(output_dir, file))
    
    # Process weight files
    file_list = os.listdir(source_dir)
    weight_map = {}
    total_params_kept = 0
    total_params_original = 0
    
    for file in file_list:
        if file.endswith("safetensors"):
            print(f"\nProcessing {file}...")
            tensors = {}
            
            with safe_open(os.path.join(source_dir, file), framework="pt", device='cpu') as f:
                for k in f.keys():
                    # Extract layer index from key
                    layer_idx = None
                    if 'layers.' in k:
                        # Extract layer number from key like "model.layers.0...."
                        parts = k.split('.')
                        for i, part in enumerate(parts):
                            if part == 'layers' and i + 1 < len(parts):
                                try:
                                    layer_idx = int(parts[i + 1])
                                    break
                                except ValueError:
                                    pass
                    
                    if 'gate' in k and layer_idx is not None:
                        # Gate tensor - select experts for this specific layer
                        experts_ids = layer_experts.get(layer_idx, default_experts)
                        current_tensors = f.get_tensor(k)
                        original_size = current_tensors.numel()
                        current_tensors = current_tensors[experts_ids]
                        kept_size = current_tensors.numel()
                        tensors[k] = current_tensors
                        weight_map[k] = file
                        total_params_original += original_size
                        total_params_kept += kept_size
                        print(f"  Gate {k}: keeping experts {experts_ids} for layer {layer_idx}")
                        
                    elif 'experts' in k and layer_idx is not None:
                        # Expert weights - keep and renumber for this specific layer
                        experts_ids = layer_experts.get(layer_idx, default_experts)
                        expert_idx = int(k.split('.')[5])  # Extract expert index
                        
                        if expert_idx in experts_ids:
                            current_tensors = f.get_tensor(k)
                            # Renumber expert to its new position
                            new_expert_idx = experts_ids.index(expert_idx)
                            new_k = k.replace(f"experts.{expert_idx}", f"experts.{new_expert_idx}")
                            tensors[new_k] = current_tensors
                            weight_map[new_k] = file
                            total_params_kept += current_tensors.numel()
                            # print(f"  Expert {k} -> {new_k} (layer {layer_idx})")
                        else:
                            # Skip this expert as it's being pruned
                            current_tensors = f.get_tensor(k)
                            total_params_original += current_tensors.numel()
                            print(f"  Pruning expert {expert_idx} from layer {layer_idx}")
                            
                    else:
                        # Other tensors - keep as is
                        current_tensors = f.get_tensor(k)
                        tensors[k] = current_tensors
                        weight_map[k] = file
                        total_params_kept += current_tensors.numel()
                        total_params_original += current_tensors.numel()
            
            # Save processed tensors
            save_file(tensors, os.path.join(output_dir, file), metadata={"format": "pt"})
            print(f"  Saved {file} with {len(tensors)} tensors")
    
    # Save weight map
    with open(os.path.join(output_dir, "model.safetensors.index.json"), 'w') as f:
        json.dump({
            "metadata": {
                "total_size": total_params_kept * 4  # Assuming float32
            },
            "weight_map": weight_map
        }, f, indent=2)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("PRUNING SUMMARY")
    print("="*50)
    print(f"Total parameters kept: {total_params_kept:,}")
    if total_params_original > 0:
        reduction = (1 - total_params_kept / total_params_original) * 100
        print(f"Parameter reduction: {reduction:.2f}%")
    
    print("\nPer-layer expert count:")
    expert_counts = {}
    for layer_idx, experts in layer_experts.items():
        count = len(experts)
        if count not in expert_counts:
            expert_counts[count] = []
        expert_counts[count].append(layer_idx)
    
    for count in sorted(expert_counts.keys()):
        layers = expert_counts[count]
        print(f"  {count} experts: layers {layers[:5]}{'...' if len(layers) > 5 else ''} ({len(layers)} layers)")
    
    print(f"\nOutput saved to: {output_dir}")
    print(f"Expert configuration saved to: {os.path.join(output_dir, 'expert_config.json')}")


def create_example_config(output_file="expert_config_example.json", num_layers=32):
    """
    Create an example configuration file for per-layer expert selection.
    
    This creates a gradual pruning strategy where early layers keep more experts
    and later layers keep fewer experts.
    """
    config = {}
    
    # Example: Gradual reduction in experts from early to later layers
    for i in range(num_layers):
        if i < 8:
            # Early layers: keep 6 experts
            config[i] = [0, 1, 2, 3, 4, 5]
        elif i < 16:
            # Middle layers: keep 4 experts
            config[i] = [0, 1, 3, 5]
        elif i < 24:
            # Later middle layers: keep 3 experts
            config[i] = [0, 2, 4]
        else:
            # Final layers: keep 2 experts
            config[i] = [0, 3]
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Example configuration saved to {output_file}")
    return config


if __name__ == '__main__':
    fire.Fire({
        'prune': select_experts_per_layer,
        'create_example': create_example_config
    })
