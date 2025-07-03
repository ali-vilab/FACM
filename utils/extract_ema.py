#!/usr/bin/env python3
"""
Extract EMA weights from PyTorch checkpoint files
Supports both standard EMA weights and PostHocEMA weights (extracts emas.ema_models[1])

Usage:
    python extract_ema.py <input_pt_file> [output_pt_file]
    
Examples:
    python extract_ema.py model.pt
    python extract_ema.py model.pt model_ema_only.pt
    python extract_ema.py /path/to/checkpoint.pt /path/to/ema_weights.pt
"""

import torch
import argparse
import os
import sys
from pathlib import Path

def extract_ema_weights(input_path, output_path=None):
    """
    Extract EMA weights from checkpoint file and save
    Supports both standard EMA and PostHocEMA weights
    
    Args:
        input_path (str): Input pt file path
        output_path (str, optional): Output file path, auto-generated if None
    
    Returns:
        str: Output file path
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading checkpoint file: {input_path}")
    try:
        # Load checkpoint
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load file: {e}")
    
    # Check checkpoint content
    ema_weights = None
    checkpoint_type = None
    
    if 'emas' in checkpoint:
        # PostHocEMA case
        checkpoint_type = "PostHocEMA"
        print("Detected PostHocEMA weights")
        
        try:
            # Try to extract ema_models[1] weights from emas
            emas_state = checkpoint['emas']
            
            # First check if it's nested structure
            if 'ema_models' in emas_state and isinstance(emas_state['ema_models'], list):
                # Nested structure
                ema_models = emas_state['ema_models']
                if len(ema_models) > 1:
                    # Extract ema_models[1] weights
                    ema_weights = ema_models[1]
                    print(f"Successfully extracted PostHocEMA ema_models[1] weights (nested), {len(ema_weights)} params")
                else:
                    raise ValueError("Not enough EMA models in emas_models (need at least 2)")
            else:
                # Flattened structure - check for ema_models.1. keys
                ema_models_1_keys = [k for k in emas_state.keys() if k.startswith('ema_models.1.ema_model.')]
                
                if ema_models_1_keys:
                    print(f"Detected flattened PostHocEMA structure, found {len(ema_models_1_keys)} ema_models.1 weights")
                    
                    # Extract and reconstruct ema_models.1.ema_model.* weights
                    ema_weights = {}
                    for key in ema_models_1_keys:
                        # Remove 'ema_models.1.ema_model.' prefix to get standard weight key
                        new_key = key.replace('ema_models.1.ema_model.', '')
                        ema_weights[new_key] = emas_state[key]
                    
                    print(f"Successfully extracted and reconstructed PostHocEMA ema_models.1 weights, {len(ema_weights)} params")
                    
                else:
                    # Check if there are at least ema_models.0 weights as fallback
                    ema_models_0_keys = [k for k in emas_state.keys() if k.startswith('ema_models.0.ema_model.')]
                    
                    if ema_models_0_keys:
                        print(f"Warning: No ema_models.1 weights found, but found {len(ema_models_0_keys)} ema_models.0 weights")
                        response = input("Use ema_models.0 weights instead? (y/N): ")
                        
                        if response.lower() in ['y', 'yes']:
                            # Extract and reconstruct ema_models.0.ema_model.* weights
                            ema_weights = {}
                            for key in ema_models_0_keys:
                                # Remove 'ema_models.0.ema_model.' prefix
                                new_key = key.replace('ema_models.0.ema_model.', '')
                                ema_weights[new_key] = emas_state[key]
                            
                            print(f"Successfully extracted and reconstructed PostHocEMA ema_models.0 weights, {len(ema_weights)} params")
                            checkpoint_type = "PostHocEMA (ema_models.0)"
                        else:
                            raise ValueError("User chose not to use ema_models.0 weights")
                    else:
                        # Print available keys for debugging
                        print(f"Available emas state keys: {list(emas_state.keys())[:10]}...")
                        raise ValueError("Cannot find PostHocEMA model weights in emas state")
                    
        except Exception as e:
            raise RuntimeError(f"Error processing PostHocEMA weights: {e}")
            
    elif 'ema' in checkpoint:
        # Standard EMA case
        checkpoint_type = "Standard EMA"
        ema_weights = checkpoint['ema']
        print(f"Detected standard EMA weights, {len(ema_weights)} params")
    else:
        raise ValueError("No 'ema' or 'emas' weights found in checkpoint file")
    
    # Generate output file path
    if output_path is None:
        input_file = Path(input_path)
        if checkpoint_type.startswith("PostHocEMA"):
            output_path = input_file.parent / f"{input_file.stem}_posthoc_ema{input_file.suffix}"
        else:
            output_path = input_file.parent / f"{input_file.stem}_ema_only{input_file.suffix}"
    
    # Create new checkpoint with only EMA weights
    ema_only_checkpoint = {
        'ema': ema_weights
    }
    
    # Add marker to indicate extraction source
    ema_only_checkpoint['_extracted_from'] = checkpoint_type
    
    # Save EMA weights
    print(f"Saving EMA weights to: {output_path}")
    try:
        torch.save(ema_only_checkpoint, output_path)
        print(f"Successfully saved EMA weights file: {output_path}")
        print(f"Weight type: {checkpoint_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to save file: {e}")
    
    # Show file size comparison
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Original file size: {original_size:.2f} MB")
    print(f"New file size: {new_size:.2f} MB")
    print(f"Compression ratio: {(1 - new_size/original_size)*100:.1f}%")
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Extract EMA weights from PyTorch checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_ema_weights.py model.pt
  python extract_ema_weights.py model.pt model_ema_only.pt
  python extract_ema_weights.py /path/to/checkpoint.pt /path/to/ema_weights.pt
        """
    )
    
    parser.add_argument('input_file', help='Input pt file path')
    parser.add_argument('output_file', nargs='?', default=None, 
                       help='Output file path (optional, auto-generated by default)')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='Force overwrite if output file exists')
    
    args = parser.parse_args()
    
    try:
        # Check if output file already exists
        if args.output_file and os.path.exists(args.output_file) and not args.force:
            response = input(f"Output file {args.output_file} exists, overwrite? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Operation cancelled")
                return
        
        # Extract EMA weights
        output_path = extract_ema_weights(args.input_file, args.output_file)
        print(f"\n EMA weights saved to: {output_path}")
        
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 