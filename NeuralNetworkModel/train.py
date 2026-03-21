"""
due to some quirks regarding the ipynb (specifically batch workers), essentially copy pasted the notebook but in py format.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


SCRIPT_DIR   = Path(__file__).parent
SPLIT_ROOT   = SCRIPT_DIR / ".." / "notebook" / "datasets" / "dataset"
LABELS_CACHE = SPLIT_ROOT / "split_labels_cache.csv"
WEIGHTS_DIR  = SCRIPT_DIR / ".." / "weights"


class FractureDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img   = Image.open(self.paths[index]).convert("RGB")
        img   = self.transform(img)
        label = self.labels[index]
        return img, label


def build_split_cache(split_root: Path, output_csv: Path) -> pl.DataFrame:
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    rows = []
    for split_name in ["train", "valid", "test"]:
        images_dir = split_root / split_name / "images"
        labels_dir = split_root / split_name / "labels"
        for label_file in sorted(labels_dir.glob("*.txt")):
            stem = label_file.stem
            image_path = next(
                (images_dir / f"{stem}{ext}" for ext in image_exts
                 if (images_dir / f"{stem}{ext}").exists()),
                None,
            )
            if image_path is None:
                continue
            label = 1 if label_file.read_text(encoding="utf-8").strip() else 0
            rows.append({"split": split_name, "image_id": stem,
                         "paths": str(image_path), "fractured": label})
    df = pl.DataFrame(rows)
    df.write_csv(output_csv)
    return df


def train_one_stage(model, train_loader, val_loader, optimizer, criterion,
                    device, num_epochs, val_threshold=0.40, log_every=20):
    accuracy  = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall    = BinaryRecall().to(device)
    f1        = BinaryF1Score().to(device)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {k: [] for k in [
        "train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
        "val_loss",   "val_acc",   "val_precision",   "val_recall",   "val_f1",
    ]}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        accuracy.reset(); precision.reset(); recall.reset(); f1.reset()

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images      = images.to(device)
            labels      = labels.to(device)
            labels_float = labels.float()
            labels_int   = labels.long()

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
                outputs = model(images).flatten()
                loss    = criterion(outputs, labels_float)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= val_threshold).long()
            accuracy.update(preds, labels_int)
            precision.update(preds, labels_int)
            recall.update(preds, labels_int)
            f1.update(preds, labels_int)

            total = len(train_loader)
            if batch_idx % log_every == 0 or batch_idx == total:
                print(f"  Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx}/{total}] "
                      f"Loss: {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        accuracy.reset(); precision.reset(); recall.reset(); f1.reset()

        with torch.no_grad():
            for images, labels in val_loader:
                images      = images.to(device)
                labels      = labels.to(device)
                labels_float = labels.float()
                labels_int   = labels.long()

                with torch.autocast(device_type=device.type,
                                    enabled=(device.type == "cuda")):
                    outputs = model(images).flatten()
                    loss    = criterion(outputs, labels_float)

                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) >= val_threshold).long()
                accuracy.update(preds, labels_int)
                precision.update(preds, labels_int)
                recall.update(preds, labels_int)
                f1.update(preds, labels_int)

        val_loss /= len(val_loader.dataset)

        t_acc = accuracy.compute().item() if False else None  # computed below
        # recompute for logging
        train_acc  = history["train_acc"][-1]  if history["train_acc"]  else 0
        # actually track properly
        for key, val in [
            ("train_loss", train_loss),
            ("train_acc",  accuracy.compute().item()),
            ("train_precision", precision.compute().item()),
            ("train_recall",    recall.compute().item()),
            ("train_f1",        f1.compute().item()),
        ]:
            history[key].append(val)

        accuracy.reset(); precision.reset(); recall.reset(); f1.reset()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device); labels = labels.to(device).long()
                with torch.autocast(device_type=device.type,
                                    enabled=(device.type == "cuda")):
                    outputs = model(images).flatten()
                preds = (torch.sigmoid(outputs) >= val_threshold).long()
                accuracy.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1.update(preds, labels)

        for key, val in [
            ("val_loss",      val_loss),
            ("val_acc",       accuracy.compute().item()),
            ("val_precision", precision.compute().item()),
            ("val_recall",    recall.compute().item()),
            ("val_f1",        f1.compute().item()),
        ]:
            history[key].append(val)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {history['train_loss'][-1]:.4f} "
            f"Acc: {history['train_acc'][-1]:.4f} "
            f"Rec: {history['train_recall'][-1]:.4f} "
            f"F1: {history['train_f1'][-1]:.4f} | "
            f"Val Loss: {history['val_loss'][-1]:.4f} "
            f"Acc: {history['val_acc'][-1]:.4f} "
            f"Rec: {history['val_recall'][-1]:.4f} "
            f"F1: {history['val_f1'][-1]:.4f}"
        )

    return history


def evaluate(model, test_loader, device, threshold, checkpoint_name):
    accuracy  = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall    = BinaryRecall().to(device)
    f1        = BinaryF1Score().to(device)
    all_labels, all_preds = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device); labels = labels.to(device).long()
            preds  = (torch.sigmoid(model(images).flatten()) >= threshold).long()
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print("\n" + "="*55)
    print("TEST RESULTS —", checkpoint_name)
    print("="*55)
    print(f"Accuracy  : {accuracy.compute().item():.4f}")
    print(f"Precision : {precision.compute().item():.4f}")
    print(f"Recall    : {recall.compute().item():.4f}")
    print(f"F1-score  : {f1.compute().item():.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=["Non-fractured", "Fractured"]))

    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-fractured", "Fractured"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {checkpoint_name} (thr={threshold})")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--workers",     type=int,   default=4,
                        help="DataLoader num_workers (default 4; safe as .py script)")
    parser.add_argument("--threshold",   type=float, default=0.40)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    WEIGHTS_DIR.mkdir(exist_ok=True)

    split_root   = SPLIT_ROOT.resolve()
    labels_cache = LABELS_CACHE.resolve()
    cache_df = pl.read_csv(labels_cache) if labels_cache.exists() \
               else build_split_cache(split_root, labels_cache)

    def get_split(name):
        df = cache_df.filter(pl.col("split") == name)
        return df["paths"].to_list(), df["fractured"].to_list()

    train_paths,  train_labels  = get_split("train")
    val_paths,    val_labels    = get_split("valid")
    test_paths,   test_labels   = get_split("test")

    n_frac     = sum(train_labels)
    n_nonfrac  = len(train_labels) - n_frac
    print(f"Train  : {len(train_labels):5d} images | {n_frac} fractured | {n_nonfrac} non-fractured")
    print(f"Valid  : {len(val_labels):5d} images")
    print(f"Test   : {len(test_labels):5d} images")

    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def make_loader(paths, labels, shuffle):
        return DataLoader(
            FractureDataset(paths, labels, transform),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.workers > 0),
        )

    train_loader = make_loader(train_paths, train_labels, shuffle=True)
    val_loader   = make_loader(val_paths,   val_labels,   shuffle=False)
    test_loader  = make_loader(test_paths,  test_labels,  shuffle=False)

    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.DEFAULT
    )
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[0].in_features, 1),
    )
    model = model.to(device)

    pos_weight = torch.tensor([n_nonfrac / n_frac], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight : {pos_weight.item():.4f}")


    if not args.skip_stage1:
        print("\n" + "="*55)
        print("STAGE 1 — Classifier head (10 epochs)")
        print("="*55)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
        train_one_stage(model, train_loader, val_loader, optimizer, criterion,
                        device, num_epochs=30, val_threshold=args.threshold)
        ckpt1 = WEIGHTS_DIR / "finetunedNN.pth"
        torch.save(model.state_dict(), ckpt1)
        print(f"Saved: {ckpt1}")

    if not args.skip_stage2:
        print("\n" + "="*55)
        print("STAGE 2 — + features[-1] (20 epochs)")
        print("="*55)
        for param in model.parameters():
            param.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        for p in model.features[-1].parameters():
            p.requires_grad = True
        model = model.to(device)

        optimizer = torch.optim.Adam([
            {"params": model.classifier.parameters(),  "lr": 1e-3},
            {"params": model.features[-1].parameters(), "lr": 1e-4},
        ])
        train_one_stage(model, train_loader, val_loader, optimizer, criterion,
                        device, num_epochs=30, val_threshold=args.threshold)
        ckpt2 = WEIGHTS_DIR / "finetunedNN_2.0.pth"
        torch.save(model.state_dict(), ckpt2)
        print(f"Saved: {ckpt2}")

    if not args.skip_stage3:
        print("\n" + "="*55)
        print("STAGE 3 — + features[-2] (20 epochs)")
        print("="*55)
        for param in model.parameters():
            param.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        for p in model.features[-1].parameters():
            p.requires_grad = True
        for p in model.features[-2].parameters():
            p.requires_grad = True
        model = model.to(device)

        optimizer = torch.optim.Adam([
            {"params": model.classifier.parameters(),   "lr": 1e-3},
            {"params": model.features[-1].parameters(), "lr": 1e-4},
            {"params": model.features[-2].parameters(), "lr": 1e-5},
        ])
        train_one_stage(model, train_loader, val_loader, optimizer, criterion,
                        device, num_epochs=20, val_threshold=args.threshold)
        ckpt3 = WEIGHTS_DIR / "finetunedNN_3.0.pth"
        torch.save(model.state_dict(), ckpt3)
        print(f"Saved: {ckpt3}")

    print("\n" + "="*55)
    print("FINAL EVALUATION ON TEST SET")
    print("="*55)
    evaluate(model, test_loader, device, args.threshold, "finetunedNN_3.0.pth")


if __name__ == "__main__":
    main()
