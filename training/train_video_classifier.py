"""
AccidentAI — LSTM Video Classifier Training Script
Trains a bidirectional LSTM on pre-extracted per-frame feature vectors.

Dataset structure expected:
  data/video_features/
    accident/       *.npy     (shape: [N_frames, 6])
    no_accident/    *.npy     (shape: [N_frames, 6])

Feature dimensions (per frame):
  [vehicle_count_norm, min_distance_norm, max_iou, avg_velocity_norm,
   trajectory_score, frame_idx_norm]

Usage:
  python train_video_classifier.py --data_dir data/video_features --epochs 50
"""

import os
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix


class VideoFeatureDataset(Dataset):
    """Loads .npy feature files for accident/no_accident classification."""

    def __init__(self, root_dir, window_size=16):
        self.samples = []
        self.window_size = window_size

        for label_idx, label_name in enumerate(["no_accident", "accident"]):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                print(f"[WARNING] Missing directory: {label_dir}")
                continue

            for fname in os.listdir(label_dir):
                if fname.endswith(".npy") or fname.endswith(".npz"):
                    self.samples.append((os.path.join(label_dir, fname), label_idx))

        random.shuffle(self.samples)
        print(f"   Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if path.endswith('.npz'):
            data = np.load(path)
            base = data['features'].astype(np.float32)  # (N, 6)
            # If ViT features present, concatenate along feature axis
            if 'vit_logits' in data and 'vit_emb' in data:
                logits = data['vit_logits'].astype(np.float32)
                emb    = data['vit_emb'].astype(np.float32)
                # Ensure same temporal length
                L = base.shape[0]
                if logits.shape[0] != L:
                    logits = np.resize(logits, (L, logits.shape[1]))
                if emb.shape[0] != L:
                    emb = np.resize(emb, (L, emb.shape[1]))
                features = np.concatenate([base, logits, emb], axis=1)
            else:
                features = base
        else:
            features = np.load(path).astype(np.float32)

        # Pad or truncate to window_size
        if len(features) < self.window_size:
            pad = np.zeros((self.window_size - len(features), features.shape[1]), dtype=np.float32)
            features = np.concatenate([pad, features], axis=0)
        elif len(features) > self.window_size:
            # Take last window_size frames
            features = features[-self.window_size:]

        return torch.tensor(features), torch.tensor(label, dtype=torch.long)


class AccidentLSTM(nn.Module):
    def __init__(self, feature_dim=6, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        last = hn[-1]
        last = self.dropout(last)
        return self.classifier(last)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load full dataset
    full_dataset = VideoFeatureDataset(args.data_dir, window_size=args.window_size)

    if len(full_dataset) == 0:
        print("❌ No data found! Please place .npy files in data/video_features/{accident,no_accident}/")
        return

    # 80/20 split
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val   = n - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    print(f"📊 Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # Infer feature dimension from a sample
    sample_feat, _ = full_dataset[0]
    feature_dim = sample_feat.shape[1]
    model = AccidentLSTM(feature_dim=feature_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * features.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total   += features.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss    += loss.item() * features.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += features.size(0)
                all_preds.extend(outputs.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        train_acc = train_correct / max(train_total, 1) * 100
        val_acc   = val_correct / max(val_total, 1) * 100
        elapsed   = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss/max(train_total,1):.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss/max(val_total,1):.4f} Acc: {val_acc:.1f}% | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"   ✅ Best model saved → {args.output}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"⏹️  Early stopping at epoch {epoch+1}")
            break

    print("\n" + "="*60)
    print(f"Best Val Accuracy: {best_acc:.2f}%")
    if all_labels:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["No Accident", "Accident"], digits=3))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM video classifier")
    parser.add_argument("--data_dir", type=str, default="data/video_features")
    parser.add_argument("--output", type=str, default="../backend/weights/video_classifier.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    train(args)
