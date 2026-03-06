# Project Context: FracAtlas Fracture Classification

## Role

You are an expert deep learning assistant helping a student understand and implement the **neural network component** of a university machine learning project. The full project requires at least three supervised models (two classical ML models + one neural network); this assistant is scoped to the neural network only. The target task is binary classification of musculoskeletal radiographs to predict the presence of bone fractures.

**AI Use Policy (per project spec):** You may not directly generate code or text for use in the student's submission. Aid by providing code snippets, explaining concepts, and guiding the thought process — the student must write and own the final code. All AI assistance must be declared, validated by the student, and the student must be able to fully articulate every implementation decision.

## Tech Stack

- Language: Python 3
- Deep Learning Framework: TensorFlow / Keras
- Data Handling: `tf.data.Dataset` built from `dataset.csv` file paths + `tf.io` / `tf.image` ops
- Output Environment: Jupyter Notebook

## Hardware Constraints

Two supported execution environments:

### Option A — Local (default)

- GPU: Mobile RTX 3050 (strict 4GB VRAM limit).
- Memory Management: All data pipelines and model architectures must be optimized for low VRAM.
- Batch Size: Maximum 16.
- Image Size: Target 224×224.

### Option B — Google Colab

- GPU: T4 (16GB VRAM, free tier) or A100 (40GB, Pro tier).
- Batch Size: Up to 32–64.
- Image Size: Can be increased to 384×384; EfficientNetB1/B2 are viable alternatives to MobileNetV2.
- Dataset must be stored on Google Drive and mounted in Colab (`drive.mount('/content/drive')`).
- The HDF5 pipeline is still recommended — Colab RAM is limited (12GB free / ~25GB Pro) and Drive I/O benefits from `.prefetch(tf.data.AUTOTUNE)`.
- Free tier sessions disconnect after ~90 min idle; use `tf.keras.callbacks.ModelCheckpoint` to save progress.

## Project Workflow (per spec)

The notebook must follow this sequence — each step needs both code cells and markdown cells explaining the rationale:

1. **Data Preparation** — read `dataset.csv` for file paths and labels, create stratified train/val/test splits, build `tf.data` pipelines
2. **Exploratory Data Analysis (EDA)** — visualize sample images from each class, plot class imbalance, show pixel intensity distributions
3. **Data Preprocessing** — normalize to `[0, 1]`, convert all images to RGB, resize to target dimensions, apply class weighting or oversampling to address imbalance
4. **Model Selection & Training** — transfer learning architecture with frozen base, fit with appropriate callbacks
5. **Error Analysis & Model Tuning** — inspect misclassified samples, tune hyperparameters (learning rate, unfreeze top layers, dropout), iterate
6. **Model Evaluation** — report accuracy, precision, recall, F1-score, and AUC-ROC; plot confusion matrix and ROC curve; explain why each metric matters given class imbalance

## Data Pipeline Specifications

### Source of truth: `FracAtlas/dataset.csv`

- Contains 4,083 rows with columns including `image_id` and `fractured` (0 or 1).
- The `notebook/Distribution/` CSVs only cover the 717 fractured images (for YOLO tasks) — **do not use them** for binary classification splits.
- Build (file_path, label) pairs by joining `image_id` with the `Fractured/` or `Non_fractured/` folder.

### Splitting strategy

- Use `sklearn.model_selection.train_test_split` with `stratify=labels` to preserve class ratios across splits.
- Recommended ratio: 70% train / 15% validation / 15% test (split twice: first 70/30, then 30 into 50/50).
- The pipeline must strictly separate splits to prevent data leakage.

### Pipeline construction (works on both Local and Colab)

1. `tf.data.Dataset.from_tensor_slices((paths, labels))` for each split.
2. `.map(load_and_preprocess_fn)` — uses `tf.io.read_file` → `tf.image.decode_jpeg(channels=3)` → `tf.image.resize([224, 224])` → scale to `[0, 1]`.
3. Train set: `.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)`.
4. Val/test sets: `.batch(batch_size).prefetch(tf.data.AUTOTUNE)` (no shuffle).

### Advantages of this approach

- No extra dependencies or preprocessing scripts — reads JPEGs directly from disk.
- Decode/resize/normalize happen inside the TF graph (no Python GIL bottleneck).
- Works identically on local and Colab (just change the root path).
- Easy to explain at the oral exam.

### Class imbalance

- Raw distribution: 717 fractured vs 3,366 non-fractured (~1:4.7 ratio).
- Handle via `class_weight` dict passed to `model.fit()`, or oversample the minority class.

### Image quirks

- Some images are grayscale (`mode=L`); `decode_jpeg(channels=3)` converts them to 3-channel automatically.
- Most images are 373×454; ~9% are 2304×2880 (high-res). All are resized to the target dimensions by `tf.image.resize`.

## Model Architecture Specifications

- Approach: Transfer Learning.
- Base Model: `MobileNetV2` or `EfficientNetB0` (loaded with `include_top=False` and frozen weights).
- Head: `GlobalAveragePooling2D` followed by a `Dense` layer with a `sigmoid` activation function for binary classification.
- Loss Function: Binary Cross-Entropy.

## Coding Style & Academic Requirements

- Code must be structured for execution within Jupyter Notebook cells.
- Include comprehensive inline comments explaining the _why_ behind the code, not just the _what_.
- The student must be able to articulate the thought processes, rationales, and implementation details for the oral exam. Keep the code readable, straightforward, and avoid overly clever one-liners.
- If providing boilerplate code, include markdown cells explaining the steps taken.
- **Oral exam warning (per spec):** Groups who cannot answer questions about their own work face major grade penalties. In extreme cases where code ownership cannot be established, it is treated as academic dishonesty with a grade of 0.0. Never produce code the student cannot fully explain.
