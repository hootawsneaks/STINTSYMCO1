# Project Details for group members:
I hope you guys have good pc's

## In order to run this project: 

### 0. run the RUNTHIS_WINDOWS.bat
### 1. Download the FracAtlas Dataset (https://figshare.com/articles/dataset/The_dataset/22363012?file=43283628)
### 2. Create the following directory structure: 
```
notebook/
├── datasets/
│   └── dataset/          # Will be created by the scripts
├── Distribution/         # Contains train/valid/test CSV splits
└── [scripts and notebooks]
```
### 3. Put the FracAtlas Dataset in the root directory (outside notebook folder)
### 4. Copy FracAtlas/images/Fractured and Non_fractured into ./notebook/images 
### 5. Run augment_with_labels.py (python augment_with_labels.py --augmentations-per-image 3)
### 6. Run organize_dataset.py (python organize_dataset.py)


## Project Scripts

### Data Organization
- **`organize_dataset.py`** - Organizes dataset into YOLO-compatible structure

### Data Augmentation  
- **`augment_with_labels.py`** - Augments fractured images with label transformations
- **`fractured_augmentation.ipynb`** - Jupyter notebook for augmentation exploration

### Training
- **`Train_8s.ipynb`** - YOLOv8s localization (detection) training
- **`Train_8s-seg_Segmentation.ipynb`** - YOLOv8s-seg segmentation training (requires mask data)

## Usage Guide

### Step 1: Organize the Dataset
```bash
python organize_dataset.py
```
This creates:
```
datasets/dataset/
├── data.yaml            # Dataset configuration
├── train/
│   ├── images/          # Training images
│   └── labels/          # Training labels (YOLO format)
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels
└── test/
    ├── images/          # Test images
    └── labels/          # Test labels
```

### Step 2: Augment Data (Optional but Recommended)
To address class imbalance (4.69:1 non-fractured:fractured):
```bash
python augment_with_labels.py --augmentations-per-image 3
```
- Creates 3 augmented versions per fractured training image
- Transforms bounding boxes for geometric augmentations
- Uses realistic X-ray augmentations (no artificial-looking transforms)

### Step 3: Train the Model

#### Option A: Detection (Localization) - Uses bounding boxes
```bash
# In Jupyter notebook or Python script:
!yolo mode=train model=yolov8s.pt data=datasets/dataset/data.yaml epochs=30 imgsz=600

# For Apple Silicon MPS acceleration:
!yolo mode=train model=yolov8s.pt data=datasets/dataset/data.yaml epochs=30 imgsz=600 device=mps
```

### Step 4: Evaluate and Predict
```bash
# Validate on test set
!yolo mode=val model=runs/detect/train/weights/best.pt data=datasets/dataset/data.yaml

# Run predictions
!yolo mode=predict model=runs/detect/train/weights/best.pt source=datasets/dataset/test/images
```

## Training Performance

### Expected Results
- **Original dataset**: 717 fractured, 3,366 non-fractured images (4.69:1 ratio)
- **After augmentation**: ~2,439 total fractured images (1.38:1 ratio)
- **Training time**: 2-8 hours depending on hardware
- **Target mAP@0.5**: > 0.85 for good fracture detection

### Monitoring Training
Watch these key metrics:
- **box_loss**: Should drop from ~3.0 to < 0.1 (bounding box accuracy)
- **cls_loss**: Should drop from ~1.5 to < 0.05 (classification accuracy)  
- **mAP@0.5**: Mean Average Precision at IoU=0.5 (target > 0.85)
- **Precision/Recall**: Balance between false positives and false negatives

## Data Augmentation Details

The augmentation pipeline uses **realistic X-ray transformations**:
- Horizontal flips (30% probability)
- Small rotations (±10°, 30% probability)
- Subtle brightness/contrast adjustments (40% probability)
- Extremely subtle Gaussian noise (0.5-1% of max, 20% probability)
- Minimal Gaussian blur (10% probability)
- Very mild elastic transforms (5% probability)
- Subtle gamma adjustment (±5%, 20% probability)
- CLAHE enhancement (15% probability)

**All transforms preserve medical realism** - no artificial-looking "TV static" noise.

## Troubleshooting

### Common Issues

1. **"IndexError: index is out of bounds for dimension with size 0"**
   - **Cause**: Using segmentation model (`yolov8s-seg.pt`) with bounding box data
   - **Fix**: Use detection model (`yolov8s.pt`) for localization

2. **Slow training on CPU**
   - **Fix**: Add `device=mps` for Apple Silicon or `device=0` for NVIDIA GPU
   - **Alternative**: Reduce `imgsz` to 320 or 416 for faster training

3. **Memory errors**
   - **Fix**: Reduce `batch` size or `imgsz`
   - **Alternative**: Use gradient accumulation

4. **Poor validation metrics**
   - **Check**: Class imbalance - run augmentation
   - **Check**: Overfitting - add more regularization
   - **Check**: Learning rate - may be too high/low

## Project Structure
```
notebook/
├── datasets/                    # Dataset organization
│   ├── dataset/                # YOLO-formatted dataset
│   ├── images/                 # Source images
│   └── labels/                 # Source labels
├── Distribution/               # CSV splits
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── runs/                       # Training outputs (auto-created)
├── organize_dataset.py         # Main organization script
├── augment_with_labels.py      # Augmentation with labels
├── fractured_augmentation.ipynb # Augmentation exploration
├── Train_8s.ipynb             # Detection training
├── Train_8s-seg_Segmentation.ipynb # Segmentation training
├── Prediction_8s.ipynb        # Model prediction
├── data.yaml                  # Dataset config
├── DATASET_SETUP.md           # Setup instructions
└── AUGMENTATION_README.md     # Augmentation guide
```

## Results Interpretation

### Key Metrics for Medical Diagnosis
- **High Precision**: Few false positives (important to avoid unnecessary treatment)
- **High Recall**: Few false negatives (critical to not miss fractures)
- **mAP@0.5**: Overall detection accuracy at IoU threshold 0.5
- **Inference Speed**: Frames per second (important for clinical workflow)

### Expected Performance
With proper training and augmentation:
- **mAP@0.5**: 0.85-0.92
- **Precision**: 0.88-0.95  
- **Recall**: 0.82-0.90
- **Inference time**: 10-50ms per image (depending on hardware)

## License and Citation

This project uses the FracAtlas dataset. If you use this code or the dataset in your research, please cite:

```bibtex
@article{fracatlas2023,
  title={FracAtlas: A Dataset for Benchmarking Fracture Detection in X-ray Images},
  author={...},
  journal={...},
  year={2023}
}
```

## Acknowledgments
- FracAtlas dataset authors for the comprehensive fracture annotation dataset
- Ultralytics for YOLOv8 implementation
- Albumentations team for image augmentation library

