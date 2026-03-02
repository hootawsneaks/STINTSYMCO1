#!/usr/bin/env python3
"""
organize.py

Reads a CSV file containing image filenames (test split) and copies the images
from datasets/images/Fractured/ or datasets/images/Non_fractured/ to
datasets/dataset/Pred/ folder.

Usage:
  python3 organize.py [csv_file_path]
  
Default CSV path: Distribution/test.csv

Example:
  python3 organize.py                        # Uses test.csv
  python3 organize.py Distribution/valid.csv # Uses valid.csv

The script:
1. Empties the Pred folder before copying
2. Copies (not moves) the images
3. Handles images from both Fractured and Non_fractured folders
4. Warns about duplicate filenames in CSV
5. Shows progress for large batches
"""

import sys
import csv
import shutil
from pathlib import Path
from collections import Counter

def main(csv_path="Distribution/test.csv"):
    # Define paths
    fractured_dir = Path("datasets/images/Fractured")
    non_fractured_dir = Path("datasets/images/Non_fractured")
    pred_dir = Path("datasets/dataset/Pred")
    
    # Ensure pred directory exists
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Empty Pred folder
    print(f"Emptying {pred_dir}...")
    for item in pred_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    print("Pred folder cleared.")
    
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    image_filenames = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            if header and header[0] != "image_id":
                print(f"Warning: Expected header 'image_id', got '{header[0]}'")
            
            for row in reader:
                if row:  # Skip empty rows
                    image_filenames.append(row[0].strip())
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Check for duplicates
    duplicate_check = Counter(image_filenames)
    duplicates = [item for item, count in duplicate_check.items() if count > 1]
    if duplicates:
        print(f"Warning: Found {len(duplicates)} duplicate filenames in CSV (first few: {duplicates[:3]})")
    
    unique_filenames = list(dict.fromkeys(image_filenames))  # Preserve order
    print(f"Found {len(image_filenames)} image filenames in CSV ({len(unique_filenames)} unique)")
    
    # Copy images
    copied_count = 0
    not_found_count = 0
    total = len(unique_filenames)
    
    for idx, img_name in enumerate(unique_filenames, 1):
        # Try Fractured folder first, then Non_fractured
        source_path = fractured_dir / img_name
        if not source_path.exists():
            source_path = non_fractured_dir / img_name
        
        if source_path.exists():
            dest_path = pred_dir / img_name
            shutil.copy2(source_path, dest_path)  # copy with metadata
            copied_count += 1
            # Show progress for large batches
            if total > 50 and idx % 20 == 0:
                print(f"  Progress: {idx}/{total} copied")
            elif idx <= 5:  # Log first few
                print(f"  Copied: {img_name}")
            elif idx == 6 and total > 10:
                print("  ...")
        else:
            print(f"  Warning: {img_name} not found in Fractured or Non_fractured folders")
            not_found_count += 1
    
    # Summary
    print(f"\nSummary:")
    print(f"  Successfully copied: {copied_count} images")
    print(f"  Not found: {not_found_count} images")
    print(f"  Destination: {pred_dir.absolute()}")
    
    if not_found_count > 0:
        print("\nWarning: Some images were not found. Check if they exist in:")
        print(f"  - {fractured_dir.absolute()}")
        print(f"  - {non_fractured_dir.absolute()}")

if __name__ == "__main__":
    # Allow optional CSV path argument
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Distribution/test.csv"
    main(csv_path)