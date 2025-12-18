from select_experts_per_layer import select_experts_per_layer
import argparse


config1 = {0: [0, 1, 2, 3, 4, 5, 6], 1: [0, 1, 2, 3, 4, 6, 7], 2: [1, 2, 3, 4, 5, 6, 7], 3: [0, 1, 3, 4, 5, 6, 7], 4: [0, 1, 2, 3, 5, 6, 7], 5: [0, 2, 3, 4, 5, 6, 7], 6: [0, 1, 2, 3, 4, 5, 6], 7: [0, 2, 3, 4, 5, 6, 7], 8: [0, 1, 2, 4, 5, 6, 7], 9: [0, 1, 2, 4, 5, 6, 7], 10: [0, 1, 2, 3, 4, 5, 7], 11: [1, 2, 3, 4, 5, 6, 7], 12: [0, 1, 2, 3, 4, 6, 7], 13: [0, 1, 2, 3, 4, 5, 6], 14: [0, 1, 2, 3, 4, 5, 7], 15: [0, 2, 3, 4, 5, 6, 7], 16: [0, 1, 2, 4, 5, 6, 7], 17: [0, 1, 3, 4, 5, 6, 7], 18: [0, 1, 2, 3, 4, 5, 7], 19: [0, 1, 2, 3, 5, 6, 7], 20: [0, 2, 3, 4, 5, 6, 7], 21: [0, 2, 3, 4, 5, 6, 7], 22: [0, 1, 2, 3, 4, 5, 6], 23: [0, 1, 2, 3, 4, 5, 7], 24: [0, 1, 2, 3, 4, 6, 7], 25: [0, 1, 2, 3, 4, 5, 6], 26: [0, 1, 2, 3, 4, 5, 6], 27: [0, 1, 2, 3, 5, 6, 7], 28: [1, 2, 3, 4, 5, 6, 7], 29: [0, 1, 2, 3, 4, 5, 6], 30: [0, 1, 2, 3, 4, 5, 7], 31: [0, 1, 2, 3, 4, 6, 7]}

config2 = {0: [0, 1, 2, 3, 4, 6], 1: [0, 1, 2, 3, 6, 7], 2: [1, 2, 3, 4, 5, 7], 3: [0, 1, 3, 4, 5, 7], 4: [0, 2, 3, 5, 6, 7], 5: [0, 2, 3, 5, 6, 7], 6: [0, 1, 2, 3, 4, 6], 7: [0, 3, 4, 5, 6, 7], 8: [0, 1, 2, 5, 6, 7], 9: [0, 1, 2, 4, 5, 7], 10: [0, 1, 3, 4, 5, 7], 11: [1, 2, 3, 4, 5, 6], 12: [0, 1, 2, 3, 4, 6], 13: [0, 1, 2, 3, 4, 6], 14: [0, 2, 3, 4, 5, 7], 15: [0, 2, 3, 4, 6, 7], 16: [0, 1, 2, 4, 5, 7], 17: [1, 3, 4, 5, 6, 7], 18: [0, 1, 2, 3, 5, 7], 19: [0, 1, 2, 3, 6, 7], 20: [0, 2, 3, 5, 6, 7], 21: [0, 2, 4, 5, 6, 7], 22: [0, 1, 2, 3, 4, 6], 23: [0, 1, 3, 4, 5, 7], 24: [1, 2, 3, 4, 6, 7], 25: [0, 1, 2, 4, 5, 6], 26: [0, 1, 2, 4, 5, 6], 27: [0, 1, 2, 3, 5, 6], 28: [1, 2, 3, 4, 6, 7], 29: [1, 2, 3, 4, 5, 6], 30: [0, 1, 2, 3, 4, 7], 31: [1, 2, 3, 4, 6, 7]}

config3 = {0: [1, 2, 3, 4, 6], 1: [0, 1, 2, 3, 7], 2: [1, 2, 4, 5, 7], 3: [0, 1, 3, 4, 7], 4: [0, 2, 3, 5, 7], 5: [0, 2, 3, 6, 7], 6: [1, 2, 3, 4, 6], 7: [0, 3, 4, 5, 7], 8: [0, 1, 5, 6, 7], 9: [0, 1, 2, 5, 7], 10: [0, 1, 3, 4, 5], 11: [1, 2, 3, 5, 6], 12: [0, 1, 2, 3, 4], 13: [1, 2, 3, 4, 6], 14: [0, 2, 3, 4, 7], 15: [0, 3, 4, 6, 7], 16: [0, 2, 4, 5, 7], 17: [1, 3, 4, 5, 7], 18: [0, 1, 3, 5, 7], 19: [1, 2, 3, 6, 7], 20: [2, 3, 5, 6, 7], 21: [0, 2, 4, 6, 7], 22: [0, 1, 3, 4, 6], 23: [0, 3, 4, 5, 7], 24: [1, 2, 3, 4, 7], 25: [0, 1, 2, 4, 5], 26: [1, 2, 4, 5, 6], 27: [0, 2, 3, 5, 6], 28: [1, 3, 4, 6, 7], 29: [2, 3, 4, 5, 6], 30: [0, 1, 2, 3, 4], 31: [1, 2, 3, 6, 7]}

config4 = {0: [1, 2, 4, 6], 1: [0, 2, 3, 7], 2: [1, 2, 4, 5], 3: [0, 1, 3, 4], 4: [0, 2, 3, 5], 5: [0, 2, 6, 7], 6: [1, 2, 3, 4], 7: [0, 3, 4, 5], 8: [1, 5, 6, 7], 9: [0, 1, 5, 7], 10: [0, 1, 3, 5], 11: [1, 2, 5, 6], 12: [0, 1, 3, 4], 13: [2, 3, 4, 6], 14: [2, 3, 4, 7], 15: [0, 3, 4, 6], 16: [2, 4, 5, 7], 17: [3, 4, 5, 7], 18: [0, 3, 5, 7], 19: [1, 2, 3, 7], 20: [2, 3, 5, 6], 21: [0, 4, 6, 7], 22: [0, 1, 3, 6], 23: [0, 3, 4, 5], 24: [1, 2, 3, 4], 25: [1, 2, 4, 5], 26: [2, 4, 5, 6], 27: [2, 3, 5, 6], 28: [1, 3, 4, 6], 29: [2, 3, 5, 6], 30: [0, 1, 3, 4], 31: [1, 3, 6, 7]}

CONFIGS = {
    1: config1,  # Prune 1, keep 7
    2: config2,  # Prune 2, keep 6
    3: config3,  # Prune 3, keep 5
    4: config4,  # Prune 4, keep 4
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune Mixtral experts based on L2 norm")
    parser.add_argument(
        "--num_pruned", 
        type=int, 
        required=True,
        choices=[1, 2, 3, 4],
        help="Number of experts to prune per layer (1-4)"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/workspace/.hf_home/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
        help="Source directory of the original model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for pruned model (default: auto-generated based on num_pruned)"
    )
    
    args = parser.parse_args()
    
    # Select config based on number of experts to prune
    config = CONFIGS[args.num_pruned]
    experts_kept = 8 - args.num_pruned
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        output_dir = f"/workspace/model/mixtral8x7B_Instruct_pruning{args.num_pruned}experts"
    else:
        output_dir = args.output_dir
    
    print(f"=" * 60)
    print(f"Pruning Configuration:")
    print(f"  Experts to prune: {args.num_pruned}")
    print(f"  Experts to keep:  {experts_kept}")
    print(f"  Source:           {args.source_dir}")
    print(f"  Output:           {output_dir}")
    print(f"=" * 60)
    
    select_experts_per_layer_masked(
        experts_config=config,
        source_dir=args.source_dir,
        output_dir=output_dir
    )
    
    print(f"\nDone! Pruned model saved to: {output_dir}")
