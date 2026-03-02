#!/usr/bin/env python3
"""
Augment fractured X-ray images WITH corresponding YOLO label transformations.
For geometric augmentations (flips, rotations, etc.), bounding boxes are transformed accordingly.
"""

import os
import sys
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import argparse

def parse_yolo_label(label_path):
    """Parse YOLO label file into list of bounding boxes.
    
    Returns: list of [class_id, x_center, y_center, width, height]
    """
    bboxes = []
    if not label_path.exists():
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                bboxes.append([x_center, y_center, width, height, class_id])
    
    return bboxes

def write_yolo_label(label_path, bboxes):
    """Write bounding boxes to YOLO label file."""
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            # bbox format: [x_center, y_center, width, height, class_id]
            x_center, y_center, width, height, class_id = bbox
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_xray_augmentation_pipeline():
    """Create realistic X-ray augmentation pipeline with bounding box support."""
    return A.Compose([
        # Geometric transforms (small, realistic)
        A.HorizontalFlip(p=0.3),  # Left-right flip is realistic for symmetrical body parts
        A.Rotate(limit=10, p=0.3),  # Very small rotations (±10° max)
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=8, p=0.2),
        
        # Intensity/brightness variations (subtle exposure differences)
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.4),
        
        # VERY SUBTLE noise (X-rays have quantum mottle, not TV static)
        A.GaussNoise(std_range=(0.005, 0.01), p=0.2),  # Extremely subtle noise (0.5-1% of max)
        
        # Almost imperceptible blur (slight patient motion or focus)
        A.GaussianBlur(blur_limit=(1, 3), p=0.1),  # Minimal blur
        
        # Very mild elastic transform
        A.ElasticTransform(alpha=0.5, sigma=15, p=0.05),  # alpha_affine removed
        
        # X-ray specific: subtle gamma adjustment
        A.RandomGamma(gamma_limit=(95, 105), p=0.2),  # ±5% only
        
        # CLAHE for contrast enhancement (common in medical imaging)
        A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=0.15),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def main():
    parser = argparse.ArgumentParser(description="Augment fractured X-ray images with label transformations")
    parser.add_argument("--augmentations-per-image", type=int, default=3,
                       help="Number of augmented versions per image")
    parser.add_argument("--output-dir", type=str, default="datasets/images/Fractured_Aug",
                       help="Output directory for augmented images")
    parser.add_argument("--labels-dir", type=str, default="../FracAtlas/Annotations/YOLO",
                       help="Directory containing original YOLO label files")
    parser.add_argument("--output-labels-dir", type=str, default="datasets/labels/Fractured_Aug",
                       help="Output directory for augmented label files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run without saving images")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of images to process (for testing)")
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    base_dir = script_dir
    data_dir = base_dir / "datasets"
    images_dir = data_dir / "images"
    fractured_dir = images_dir / "Fractured"
    
    # Handle output directories
    if Path(args.output_dir).is_absolute():
        fractured_aug_dir = Path(args.output_dir)
    else:
        fractured_aug_dir = script_dir / args.output_dir
    
    if Path(args.labels_dir).is_absolute():
        labels_dir = Path(args.labels_dir)
    else:
        labels_dir = script_dir / args.labels_dir
    
    if Path(args.output_labels_dir).is_absolute():
        output_labels_dir = Path(args.output_labels_dir)
    else:
        output_labels_dir = script_dir / args.output_labels_dir
    
    # Create output directories
    fractured_aug_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training split
    distribution_dir = base_dir / "Distribution"
    train_csv_path = distribution_dir / "train.csv"
    print(f"Loading training split from: {train_csv_path}")
    
    # Simple CSV parsing (handles line number format from read_file)
    training_image_ids = []
    if train_csv_path.exists():
        with open(train_csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        img_id = parts[1].strip()
                        if img_id and img_id != "image_id":
                            training_image_ids.append(img_id)
                elif line and not line.startswith('|') and line != "image_id":
                    training_image_ids.append(line)
    
    print(f"Found {len(training_image_ids)} images in training set")
    
    # Get all available fractured images
    fractured_images = {}
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in fractured_dir.glob(ext):
            fractured_images[img_path.name] = img_path
    
    print(f"Found {len(fractured_images)} total fractured images")
    
    # Filter to only training set fractured images
    training_fractured = []
    for img_id in training_image_ids:
        if img_id in fractured_images:
            training_fractured.append(fractured_images[img_id])
        else:
            print(f"Warning: Training image {img_id} not found in fractured folder")
    
    # Apply limit if specified
    if args.limit > 0 and args.limit < len(training_fractured):
        training_fractured = training_fractured[:args.limit]
        print(f"Limiting to first {args.limit} images (for testing)")
    
    print(f"\nAugmenting {len(training_fractured)} fractured images from training set")
    print(f"Creating {args.augmentations_per_image} augmented versions per image")
    print(f"Total to create: {len(training_fractured) * args.augmentations_per_image}")
    print(f"Image output: {fractured_aug_dir.absolute()}")
    print(f"Label output: {output_labels_dir.absolute()}")
    
    if args.dry_run:
        print("\nDry run mode - no images/labels will be saved.")
        print("First 5 images to augment:")
        for img_path in training_fractured[:5]:
            label_path = labels_dir / f"{img_path.stem}.txt"
            has_label = label_path.exists()
            print(f"  - {img_path.name} (label: {'yes' if has_label else 'no'})")
        return
    
    # Create augmentation pipeline
    augmentation = create_xray_augmentation_pipeline()
    
    # Augment images and labels
    augmented_count = 0
    total_to_create = len(training_fractured) * args.augmentations_per_image
    
    for i, img_path in enumerate(training_fractured):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load {img_path.name}")
                continue
            
            # Convert BGR to RGB for albumentations
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load corresponding label
            label_path = labels_dir / f"{img_path.stem}.txt"
            bboxes = parse_yolo_label(label_path)
            
            # Extract class labels for albumentations
            class_labels = [bbox[4] for bbox in bboxes] if bboxes else []
            
            # Create augmented versions
            for aug_idx in range(args.augmentations_per_image):
                # Prepare data for augmentation
                data = {"image": image}
                if bboxes:
                    data["bboxes"] = bboxes
                    data["class_labels"] = class_labels
                
                # Apply augmentation
                augmented = augmentation(**data)
                aug_image = augmented['image']
                
                # Get transformed bounding boxes (if any)
                aug_bboxes = augmented.get('bboxes', [])
                
                # Generate filenames
                original_stem = img_path.stem
                aug_filename = f"{original_stem}_aug{aug_idx+1:03d}.jpg"
                aug_path = fractured_aug_dir / aug_filename
                
                # Convert back to BGR for saving
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                
                # Save augmented image
                cv2.imwrite(str(aug_path), aug_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save augmented label (if there were bounding boxes)
                if aug_bboxes:
                    aug_label_filename = f"{original_stem}_aug{aug_idx+1:03d}.txt"
                    aug_label_path = output_labels_dir / aug_label_filename
                    write_yolo_label(aug_label_path, aug_bboxes)
                elif label_path.exists():  # Original had empty label file
                    aug_label_filename = f"{original_stem}_aug{aug_idx+1:03d}.txt"
                    aug_label_path = output_labels_dir / aug_label_filename
                    # Create empty label file
                    open(aug_label_path, 'w').close()
                
                augmented_count += 1
                
                # Show progress
                if augmented_count % 50 == 0:
                    print(f"  Created {augmented_count}/{total_to_create} augmented images...")
            
            # Show progress per original image
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(training_fractured)} original images")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("AUGMENTATION WITH LABELS COMPLETE!")
    print(f"{'='*60}")
    print(f"Original fractured images in training set: {len(training_fractured)}")
    print(f"Augmented images created: {augmented_count}")
    print(f"Total output image files: {len(list(fractured_aug_dir.glob('*.jpg')))}")
    print(f"Total output label files: {len(list(output_labels_dir.glob('*.txt')))}")
    print(f"\nImage output directory: {fractured_aug_dir.absolute()}")
    print(f"Label output directory: {output_labels_dir.absolute()}")
    
    # Count all fractured images for ratio calculation
    all_fractured = len(fractured_images)
    non_fractured_dir = images_dir / "Non_fractured"
    non_fractured_count = sum(1 for _ in non_fractured_dir.glob('*.jpg')) + \
                         sum(1 for _ in non_fractured_dir.glob('*.jpeg')) + \
                         sum(1 for _ in non_fractured_dir.glob('*.png'))
    
    print(f"\n{'='*60}")
    print("CLASS BALANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Original fractured images: {all_fractured}")
    print(f"Augmented fractured images: {augmented_count}")
    print(f"Total fractured images (original + augmented): {all_fractured + augmented_count}")
    print(f"Non-fractured images: {non_fractured_count}")
    print(f"\nOriginal ratio (non-fractured:fractured): {non_fractured_count/all_fractured:.2f}:1")
    print(f"New ratio (non-fractured:total_fractured): {non_fractured_count/(all_fractured + augmented_count):.2f}:1")
    
    # Save summary file
    summary_path = fractured_aug_dir / "augmentation_summary.txt"
    summary_content = f"""# Fractured Image Augmentation with Labels Summary

## Statistics:
- Original fractured images in training set: {len(training_fractured)}
- Augmented images created: {augmented_count}
- Augmentations per image: {args.augmentations_per_image}
- Label files generated: {len(list(output_labels_dir.glob('*.txt')))}

## Directories:
- Augmented images: {fractured_aug_dir.absolute()}
- Augmented labels: {output_labels_dir.absolute()}

## Class Balance:
- Original fractured images: {all_fractured}
- Augmented fractured images: {augmented_count}
- Total fractured images: {all_fractured + augmented_count}
- Non-fractured images: {non_fractured_count}
- Original ratio: {non_fractured_count/all_fractured:.2f}:1 (non-fractured:fractured)
- New ratio: {non_fractured_count/(all_fractured + augmented_count):.2f}:1

## Augmentation Pipeline:
- Horizontal flips (30% probability) - bounding boxes transformed
- Small rotations (±10 degrees, 30% probability) - bounding boxes transformed  
- Shift/scale/rotate combinations (20% probability) - bounding boxes transformed
- Brightness/contrast adjustments (40% probability)
- Extremely subtle Gaussian noise (20% probability)
- Minimal Gaussian blur (10% probability)
- Very mild elastic transforms (5% probability)
- Subtle gamma adjustment (20% probability)
- CLAHE enhancement (15% probability)

## Notes:
- Only images from the training split were augmented
- All transforms are physically plausible for X-ray images
- Bounding boxes are transformed for geometric augmentations
- Augmented images are saved with '_augXXX' suffix
- Corresponding label files are created with same naming convention
"""
    
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()