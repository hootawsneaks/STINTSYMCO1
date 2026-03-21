#!/usr/bin/env python3

import os
import sys
import shutil
import random
from pathlib import Path
import argparse

def parse_csv(csv_path):
    """Parse CSV file, handling both regular format and line-number format."""
    image_ids = []
    if not csv_path.exists():
        return image_ids
    
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '|' in line:  
                parts = line.split('|')
                if len(parts) >= 2:
                    img_id = parts[1].strip()
                    if img_id and img_id != "image_id":
                        image_ids.append(img_id)
            elif line != "image_id": 
                image_ids.append(line)
    
    return image_ids

def find_image_source(image_name, fractured_dir, non_fractured_dir, fractured_aug_dir=None):
    """Find where an image is located."""
   
    if fractured_aug_dir:
        aug_path = fractured_aug_dir / image_name
        if aug_path.exists():
            return aug_path, "augmented"
    
    # check Fractured
    frac_path = fractured_dir / image_name
    if frac_path.exists():
        return frac_path, "fractured"
    
    # check Non_fractured
    non_frac_path = non_fractured_dir / image_name
    if non_frac_path.exists():
        return non_frac_path, "non_fractured"
    
    return None, "not_found"

def find_label_source(image_name, label_type, yolo_labels_dir, augmented_labels_dir=None):
    """Find corresponding label file."""
    label_name = image_name.rsplit('.', 1)[0] + '.txt'
    
    if label_type == "augmented" and augmented_labels_dir:
        # augmented images have labels in augmented_labels_dir
        label_path = augmented_labels_dir / label_name
        if label_path.exists():
            return label_path
        
        return None
    
   
    label_path = yolo_labels_dir / label_name
    if label_path.exists():
        return label_path
    
   
    return None

def organize_split(split_name, csv_path, output_dir, fractured_dir, non_fractured_dir,
                   yolo_labels_dir, fractured_aug_dir=None, augmented_labels_dir=None,
                   nonfrac_images=None, dry_run=False):
    """Organize a single split."""
    print(f"\n=== Organizing {split_name.upper()} split ===")
    
   
    split_images_dir = output_dir / split_name / "images"
    split_labels_dir = output_dir / split_name / "labels"
    
    if not dry_run:
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        
        for item in split_images_dir.iterdir():
            if item.is_file():
                item.unlink()
        for item in split_labels_dir.iterdir():
            if item.is_file():
                item.unlink()
    
    
    image_ids = parse_csv(csv_path)
    print(f"Found {len(image_ids)} images in CSV")
    
    if not image_ids:
        print("Warning: No images found in CSV!")
        return 0, 0
    
    
    stats = {"images": 0, "labels": 0, "errors": []}
    
    for idx, img_name in enumerate(image_ids, 1):
       
        source_path, img_type = find_image_source(
            img_name, fractured_dir, non_fractured_dir,
            fractured_aug_dir if split_name == "train" else None
        )
        
        if not source_path:
            stats["errors"].append(f"Image not found: {img_name}")
            continue
        
       
        label_source = find_label_source(
            img_name, img_type, yolo_labels_dir,
            augmented_labels_dir if split_name == "train" else None
        )
        
        
        dest_image_path = split_images_dir / img_name
        if not dry_run:
            shutil.copy2(source_path, dest_image_path)
        stats["images"] += 1
        
        
        dest_label_path = split_labels_dir / (img_name.rsplit('.', 1)[0] + '.txt')
        if label_source and not dry_run:
            shutil.copy2(label_source, dest_label_path)
            stats["labels"] += 1
        elif not dry_run:
            
            with open(dest_label_path, 'w') as f:
                pass
            stats["labels"] += 1
        else:
            stats["labels"] += 1 
        
        # Progress
        if idx % 50 == 0 or idx == len(image_ids):
            print(f"  Processed {idx}/{len(image_ids)} images")
    
    
    if split_name == "train" and fractured_aug_dir and fractured_aug_dir.exists():
        print(f"\nAdding all augmented images to training set...")
        augmented_images = list(fractured_aug_dir.glob("*.jpg"))
        print(f"Found {len(augmented_images)} augmented images")
        
        for idx, aug_path in enumerate(augmented_images, 1):
            img_name = aug_path.name
            
           
            if (split_images_dir / img_name).exists():
                continue
            
           
            label_source = None
            if augmented_labels_dir:
                label_name = img_name.rsplit('.', 1)[0] + '.txt'
                label_source = augmented_labels_dir / label_name
            
           
            dest_image_path = split_images_dir / img_name
            if not dry_run:
                shutil.copy2(aug_path, dest_image_path)
            stats["images"] += 1
            
            
            dest_label_path = split_labels_dir / (img_name.rsplit('.', 1)[0] + '.txt')
            if label_source and label_source.exists() and not dry_run:
                shutil.copy2(label_source, dest_label_path)
                stats["labels"] += 1
            elif not dry_run:
                
                with open(dest_label_path, 'w') as f:
                    pass
                stats["labels"] += 1
            else:
                stats["labels"] += 1
            
            # Progress
            if idx % 50 == 0 or idx == len(augmented_images):
                print(f"  Processed {idx}/{len(augmented_images)} augmented images")
    
    if nonfrac_images:
        print(f"\nAdding {len(nonfrac_images)} non-fractured images...")
        for idx, nf_path in enumerate(nonfrac_images, 1):
            dest_image_path = split_images_dir / nf_path.name
            dest_label_path = split_labels_dir / (nf_path.stem + '.txt')

            if not dry_run:
                shutil.copy2(nf_path, dest_image_path)
                open(dest_label_path, 'w').close()  # empty label = no fracture

            stats["images"] += 1
            stats["labels"] += 1

            if idx % 100 == 0 or idx == len(nonfrac_images):
                print(f"  Processed {idx}/{len(nonfrac_images)} non-fractured images")


    print(f"\n{split_name.upper()} split summary:")
    print(f"  Images: {stats['images']}")
    print(f"  Labels: {stats['labels']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])} (first 3: {stats['errors'][:3]})")
    
    return stats["images"], stats["labels"]

def main():
    parser = argparse.ArgumentParser(description="Organize dataset into YOLO structure")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without copying files")
    parser.add_argument("--include-augmented", action="store_true", default=True,
                       help="Include augmented images in training set (default: True)")
    parser.add_argument("--no-augmented", action="store_false", dest="include_augmented",
                       help="Exclude augmented images")
    args = parser.parse_args()
    
    
    script_dir = Path(__file__).parent
    notebook_dir = script_dir.parent / "notebook"
    data_dir = notebook_dir / "datasets"
    images_dir = data_dir / "images"
    distribution_dir = notebook_dir / "Distribution"
    yolo_labels_dir = script_dir.parent / "FracAtlas" / "Annotations" / "YOLO"
    output_dir = data_dir / "dataset"

   
    fractured_aug_dir = images_dir / "Fractured_Aug"
    augmented_labels_dir = data_dir / "labels" / "Fractured_Aug"
    
   
    has_augmented = args.include_augmented and fractured_aug_dir.exists() and any(fractured_aug_dir.glob("*.jpg"))
    
    print("="*60)
    print("DATASET ORGANIZATION")
    print("="*60)
    print(f"Notebook directory: {notebook_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Include augmented: {has_augmented}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    # Check required directories
    required_dirs = [
        (images_dir / "Fractured", "Fractured images"),
        (images_dir / "Non_fractured", "Non_fractured images"),
        (yolo_labels_dir, "YOLO labels"),
    ]
    
    for path, desc in required_dirs:
        if not path.exists():
            print(f"Error: {desc} directory not found: {path}")
            sys.exit(1)
    
    
    csv_files = {
        "train": distribution_dir / "train.csv",
        "valid": distribution_dir / "valid.csv",
        "test": distribution_dir / "test.csv",
    }
    
    for name, path in csv_files.items():
        if not path.exists():
            print(f"Error: {name} CSV not found: {path}")
            sys.exit(1)
    
   
    all_nonfrac = sorted((images_dir / "Non_fractured").glob("*.jpg")) + \
                  sorted((images_dir / "Non_fractured").glob("*.png"))
    random.seed(42)
    random.shuffle(all_nonfrac)

    n_frac_train = len(parse_csv(csv_files["train"]))
    n_frac_val   = len(parse_csv(csv_files["valid"]))
    n_frac_test  = len(parse_csv(csv_files["test"]))
    n_frac_total = n_frac_train + n_frac_val + n_frac_test
    n_nf_total   = len(all_nonfrac)

    n_nf_val   = round(n_nf_total * n_frac_val   / n_frac_total)
    n_nf_test  = round(n_nf_total * n_frac_test  / n_frac_total)
    n_nf_train = n_nf_total - n_nf_val - n_nf_test

    nonfrac_splits = {
        "train": all_nonfrac[:n_nf_train],
        "valid": all_nonfrac[n_nf_train:n_nf_train + n_nf_val],
        "test":  all_nonfrac[n_nf_train + n_nf_val:],
    }

    print(f"\nNon-fractured split: {n_nf_train} train | {n_nf_val} valid | {n_nf_test} test")

    
    total_images = 0
    total_labels = 0

    for split_name, csv_path in csv_files.items():
       
        split_aug_dir = fractured_aug_dir if (split_name == "train" and has_augmented) else None
        split_aug_labels = augmented_labels_dir if (split_name == "train" and has_augmented) else None

        images, labels = organize_split(
            split_name=split_name,
            csv_path=csv_path,
            output_dir=output_dir,
            fractured_dir=images_dir / "Fractured",
            non_fractured_dir=images_dir / "Non_fractured",
            yolo_labels_dir=yolo_labels_dir,
            fractured_aug_dir=split_aug_dir,
            augmented_labels_dir=split_aug_labels,
            nonfrac_images=nonfrac_splits[split_name],
            dry_run=args.dry_run
        )

        total_images += images
        total_labels += labels
    
    
    print(f"\n{'='*60}")
    print("ORGANIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    
    if not args.dry_run:
        print(f"\nDataset structure created at: {output_dir.absolute()}")
        
        
        print("\nFile counts:")
        for split_name in ["train", "valid", "test"]:
            split_dir = output_dir / split_name
            if split_dir.exists():
                img_dir = split_dir / "images"
                lbl_dir = split_dir / "labels"
                img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
                lbl_count = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
                print(f"  {split_name}: {img_count} images, {lbl_count} labels")
    else:
        print("\nDRY RUN - No files were copied.")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()