#!/usr/bin/env python3
"""
Test script for WHU dataset loading

This script tests the WHU dataset class to ensure it can properly load
the dataset with the actual file structure.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.stereo_datasets as datasets

def test_whu_dataset():
    """Test the WHU dataset loading"""
    
    print("Testing WHU dataset loading...")
    
    # Test parameters
    aug_params = {
        'crop_size': [320, 736],
        'min_scale': -0.2,
        'max_scale': 0.5,
        'do_flip': False
    }
    
    # Test different splits
    splits = ['training', 'validation', 'testing']
    
    for split in splits:
        print(f"\n--- Testing {split} split ---")
        
        try:
            # Create dataset
            dataset = datasets.WHU(aug_params=aug_params, image_set=split)
            print(f"✓ Successfully created {split} dataset")
            print(f"  Dataset size: {len(dataset)}")
            
            if len(dataset) > 0:
                # Test loading a sample
                try:
                    sample = dataset[0]
                    meta, left, right, disp_gt, valid = sample
                    print(f"✓ Successfully loaded sample from {split}")
                    
                    # Check sample structure
                    if len(sample) == 5:  # [index, left, right, disp_gt, valid]
                        print(f"  Left image shape: {left.shape}")
                        print(f"  Right image shape: {right.shape}")
                        print(f"  Disparity shape: {disp_gt.shape}")
                        print(f"  Valid mask shape: {valid.shape}")
                        
                        # Check data types
                        print(f"  Left image dtype: {left.dtype}")
                        print(f"  Right image dtype: {right.dtype}")
                        print(f"  Disparity dtype: {disp_gt.dtype}")
                        print(f"  Valid mask dtype: {valid.dtype}")
                        
                        # Check value ranges
                        print(f"  Left image range: [{left.min():.3f}, {left.max():.3f}]")
                        print(f"  Right image range: [{right.min():.3f}, {right.max():.3f}]")
                        print(f"  Disparity range: [{disp_gt.min():.3f}, {disp_gt.max():.3f}]")
                        print(f"  Valid mask range: [{valid.min():.3f}, {valid.max():.3f}]")
                        
                        # Check for NaN or inf values
                        if torch.isnan(left).any() or torch.isinf(left).any():
                            print("  ⚠️  Left image contains NaN or inf values")
                        if torch.isnan(right).any() or torch.isinf(right).any():
                            print("  ⚠️  Right image contains NaN or inf values")
                        if torch.isnan(disp_gt).any() or torch.isinf(disp_gt).any():
                            print("  ⚠️  Disparity contains NaN or inf values")
                        
                    else:
                        print(f"  ⚠️  Unexpected sample structure: {len(sample)} elements")
                        
                except Exception as e:
                    print(f"  ✗ Failed to load sample: {e}")
            else:
                print(f"  ⚠️  {split} dataset is empty")
                
        except Exception as e:
            print(f"  ✗ Failed to create {split} dataset: {e}")

def visualize_sample(dataset, index=0):
    """Visualize a sample from the dataset"""
    
    print(f"\n--- Visualizing sample {index} ---")
    
    try:
        sample = dataset[index]
        meta, left, right, disp_gt, valid = sample
        
        # Convert to numpy for visualization
        left_np = left.permute(1, 2, 0).numpy()
        right_np = right.permute(1, 2, 0).numpy()
        disp_np = disp_gt.squeeze().numpy()
        valid_np = valid.squeeze().numpy()
        
        # Normalize images for visualization
        left_np = (left_np - left_np.min()) / (left_np.max() - left_np.min())
        right_np = (right_np - right_np.min()) / (right_np.max() - right_np.min())
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].imshow(left_np)
        axes[0, 0].set_title('Left Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(right_np)
        axes[0, 1].set_title('Right Image')
        axes[0, 1].axis('off')
        
        disp_vis = axes[1, 0].imshow(disp_np, cmap='viridis')
        axes[1, 0].set_title('Disparity Ground Truth')
        axes[1, 0].axis('off')
        plt.colorbar(disp_vis, ax=axes[1, 0])
        
        valid_vis = axes[1, 1].imshow(valid_np, cmap='gray')
        axes[1, 1].set_title('Valid Mask')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path('temp/whu_visualization')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'whu_sample_{index}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {output_dir / f'whu_sample_{index}.png'}")
        
        plt.close()
        
    except Exception as e:
        print(f"  ✗ Failed to visualize sample: {e}")

def main():
    """Main function"""
    
    print("WHU Dataset Test Script")
    print("=" * 50)
    
    # Test dataset loading
    test_whu_dataset()
    
    # Test visualization
    try:
        aug_params = {
            'crop_size': [320, 736],
            'min_scale': -0.2,
            'max_scale': 0.5,
            'do_flip': False
        }
        
        dataset = datasets.WHU(aug_params=aug_params, image_set='training')
        
        if len(dataset) > 0:
            visualize_sample(dataset, 0)
            
            # Test multiple samples
            print(f"\n--- Testing multiple samples ---")
            for i in range(min(5, len(dataset))):
                try:
                    sample = dataset[i]
                    meta, left, right, disp_gt, valid = sample
                    print(f"  Sample {i}: ✓")
                except Exception as e:
                    print(f"  Sample {i}: ✗ ({e})")
        else:
            print("\n⚠️  Dataset is empty, skipping visualization")
            
    except Exception as e:
        print(f"\n✗ Failed to create dataset for visualization: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == '__main__':
    main() 