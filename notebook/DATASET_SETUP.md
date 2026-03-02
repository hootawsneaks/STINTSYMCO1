# Dataset Setup for YOLO Fracture Detection

This document summarizes the dataset organization and augmentation pipeline created.

## Current Status

✅ **Dataset organized into YOLO structure** (run `organize_dataset.py`):
- `datasets/dataset/train/images/` - 575 images (574 original + 1 augmented)
- `datasets/dataset/train/labels/` - 575 labels
- `datasets/dataset/valid/images/` - 82 images
- `datasets/dataset/valid/labels/` - 82 labels  
- `datasets/dataset/test/images/` - 61 images
- `datasets/dataset/test/labels/` - 61 labels
- `datasets/dataset/data.yaml` - Dataset configuration file

✅ **Augmentation script ready** (`augment_with_labels.py`):
- Creates realistic X-ray augmentations (subtle noise, flips, rotations)
- Transforms bounding boxes for geometric augmentations
- Saves both images and labels with `_augXXX` suffix

## Files Created

### Core Scripts:
1. `organize_dataset.py` - Main organization script
2. `augment_with_labels.py` - Augmentation with label transformation
3. `fractured_augmentation.ipynb` - Jupyter notebook for exploration

### Documentation:
1. `AUGMENTATION_README.md` - Augmentation instructions
2. `DATASET_SETUP.md` - This file

## Next Steps

### 1. Generate More Augmented Images (Recommended)

Current class imbalance:
- Fractured images: 717 total (575 in training set)
- Non-fractured images: 3,366 total
- Ratio: ~4.69:1 (non-fractured:fractured)

To improve balance, generate more augmented fractured images:

```bash
# Generate 3 augmented versions per training fractured image (574 × 3 = 1,722 new images)
python3 augment_with_labels.py --augmentations-per-image 3

# This will create:
# - datasets/images/Fractured_Aug/*.jpg (augmented images)
# - datasets/labels/Fractured_Aug/*.txt (transformed labels)
```

### 2. Re-organize Dataset with Augmented Images

```bash
# This will include all augmented images in training set
python3 organize_dataset.py
```

Expected result after full augmentation:
- Training fractured images: ~2,297 (575 original + 1,722 augmented)
- Total fractured images: ~2,439 (717 original + 1,722 augmented)
- New ratio: ~1.38:1 (vs original 4.69:1) - **3.4× more balanced**

### 3. Train YOLO Model

The dataset is now ready for YOLO training:

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # or 'yolov8s-seg.pt'
model.train(
    data='datasets/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## File Structure

```
notebook/
├── organize_dataset.py          # Main organization script
├── augment_with_labels.py       # Augmentation with labels
├── fractured_augmentation.ipynb # Exploration notebook
├── organize.py                  # Original test-only script
├── datasets/
│   ├── dataset/                 # YOLO dataset
│   │   ├── data.yaml            # Dataset config
│   │   ├── train/images/        # Training images
│   │   ├── train/labels/        # Training labels
│   │   ├── valid/images/        # Validation images
│   │   ├── valid/labels/        # Validation labels
│   │   ├── test/images/         # Test images
│   │   └── test/labels/         # Test labels
│   ├── images/                  # Source images
│   │   ├── Fractured/           # Original fractured images
│   │   ├── Non_fractured/       # Non-fractured images
│   │   └── Fractured_Aug/       # Augmented fractured images
│   └── labels/                  # Source labels
│       └── Fractured_Aug/       # Augmented labels
├── Distribution/                # CSV splits
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── AUGMENTATION_README.md       # Augmentation guide
```

## Augmentation Details

The augmentation pipeline uses **realistic X-ray transformations**:
- Horizontal flips (30% probability)
- Small rotations (±10°, 30% probability)
- Subtle brightness/contrast adjustments (40% probability)
- Extremely subtle Gaussian noise (0.5-1% of max, 20% probability)
- Minimal Gaussian blur (10% probability)
- Very mild elastic transforms (5% probability)
- Subtle gamma adjustment (±5%, 20% probability)
- CLAHE enhancement (15% probability)

**All transforms are physically plausible** for X-ray images (no TV static!).

## Notes

- Empty label files are created for non-fractured images (YOLO standard)
- Augmented images only added to training set (not validation/test)
- Original images are never modified - only copies are created
- The `organize.py` script is deprecated (use `organize_dataset.py` instead)

## Troubleshooting

1. **Missing labels**: Ensure `../FracAtlas/Annotations/YOLO/` contains label files
2. **Image not found**: Check `datasets/images/Fractured/` and `Non_fractured/` directories
3. **Permission errors**: Run scripts with appropriate permissions
4. **Memory issues**: For large augmentations, process in batches using `--limit` flag

## Quick Start

```bash
# 1. Generate augmented images (optional)
python3 augment_with_labels.py --augmentations-per-image 3

# 2. Organize dataset
python3 organize_dataset.py

# 3. Verify structure
ls -la datasets/dataset/
```

Your dataset is now ready for YOLO training!