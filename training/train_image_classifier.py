"""
Fine-tune tiya1012/vit-accident-image on combined accident datasets.

This script:
  1. Loads the pretrained ViT (already at F1=0.93)
  2. Loads your combined local dataset (from download_datasets.py)
  3. Fine-tunes the classification head for ~5 epochs
  4. Saves the model to backend/weights/image_classifier_vit/

Usage:
    python training/train_image_classifier.py
    python training/train_image_classifier.py --epochs 10 --lr 2e-5
    python training/train_image_classifier.py --data-dir training/data/images

Requirements:
    pip install transformers datasets torch pillow tqdm scikit-learn
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────────────

HF_MODEL_ID   = "tiya1012/vit-accident-image"
OUTPUT_DIR    = Path("backend/weights/image_classifier_vit")
DATA_DIR      = Path("training/data/images")

IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS        = 5
LR            = 2e-5
WEIGHT_DECAY  = 0.01
SEED          = 42
VAL_SPLIT     = 0.15    # used only if no val/ subdir exists


# ─── Dataset ─────────────────────────────────────────────────────────────────────

class AccidentDataset(Dataset):
    """
    Loads images from:
        data_dir/accident/    → label 0 (Accident)
        data_dir/no_accident/ → label 1 (No Accident)
    """

    LABEL_MAP = {"accident": 0, "no_accident": 1}

    def __init__(self, data_dir: Path, processor, split: str = "train"):
        self.processor = processor
        self.samples: List[Tuple[Path, int]] = []

        for class_name, label in self.LABEL_MAP.items():
            class_dir = data_dir / class_name / split
            if not class_dir.exists():
                # Try without split subfolder
                class_dir = data_dir / class_name
                if not class_dir.exists():
                    print(f"  ⚠️  Directory not found: {class_dir}")
                    continue

            exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
            for ext in exts:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label))

        random.shuffle(self.samples)
        print(f"  [{split}] Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        encoding = self.processor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(label, dtype=torch.long)


# ─── Training ────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    print(f"📂  Data:   {args.data_dir}")
    print(f"💾  Output: {args.output_dir}")

    # ── Load model and processor ─────────────────────────────────────────────────
    from transformers import ViTForImageClassification, ViTImageProcessor

    print(f"\n⬇️  Loading {HF_MODEL_ID}...")
    processor = ViTImageProcessor.from_pretrained(HF_MODEL_ID)
    model = ViTForImageClassification.from_pretrained(
        HF_MODEL_ID,
        ignore_mismatched_sizes=True,  # in case num_labels differs
    )
    model.to(device)
    print("✅ Model loaded")

    # ── Datasets ─────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    print("\n📊 Loading datasets...")
    train_ds = AccidentDataset(data_dir, processor, split="train")
    val_ds   = AccidentDataset(data_dir, processor, split="val")

    if len(train_ds) == 0:
        print("❌ No training images found!")
        print(f"   Run first: python training/download_datasets.py")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Optimizer ────────────────────────────────────────────────────────────────
    # Fine-tune: use small LR for all layers, slightly higher for the classifier head
    head_params  = [p for n, p in model.named_parameters() if "classifier" in n]
    body_params  = [p for n, p in model.named_parameters() if "classifier" not in n]

    optimizer = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr * 0.1},   # backbone: 10× smaller LR
        {"params": head_params, "lr": args.lr},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # ── Training loop ────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    output_path  = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total   = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds       = outputs.logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss/(total/args.batch_size+1):.3f}",
                             acc=f"{correct/total:.3f}")

        train_acc = correct / total
        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for pixel_values, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                outputs = model(pixel_values=pixel_values)
                preds   = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"\n  Epoch {epoch}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(output_path))
            processor.save_pretrained(str(output_path))
            print(f"  💾 Best model saved (val_acc={val_acc:.4f}) → {output_path}")

    print(f"\n🎉 Training complete! Best val_acc = {best_val_acc:.4f}")
    print(f"   Model saved to: {output_path}")
    print(f"\nNext step: Update backend/models/accident_classifier.py")
    print(f"  Change HF_MODEL_ID to: str(Path('{output_path}').absolute())")


# ─── Entry point ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ViT for accident classification")
    p.add_argument("--data-dir",   default=str(DATA_DIR),    help="Dataset directory")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),  help="Output model directory")
    p.add_argument("--epochs",     default=EPOCHS, type=int)
    p.add_argument("--lr",         default=LR,     type=float)
    p.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
