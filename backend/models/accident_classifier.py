"""
ViT-based image accident classifier using pre-trained HuggingFace model.

Primary model: tiya1012/vit-accident-image
  - Fine-tuned on CCTV accident footage
  - Binary: Accident / No Accident  (F1 = 0.93)
  - Downloads ~330 MB on first use, cached in HuggingFace cache

Fallback chain:
  1. Local fine-tuned weights (weights/image_classifier.pt)
  2. tiya1012/vit-accident-image  (HuggingFace, best accuracy)
  3. EfficientNet-B1 pretrained   (ImageNet, lower accuracy fallback)
"""

import os
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image


# ─── Labels ─────────────────────────────────────────────────────────────────────

# tiya1012/vit-accident-image uses id2label: {0: "Accident", 1: "No Accident"}
VIT_ID2LABEL = {0: "Accident", 1: "No Accident"}

# Map ViT binary output → our 3-class system for UI compatibility
VIT_TO_APP_LABEL = {
    "Accident":    "collision",
    "No Accident": "normal",
}

CLASSES = ["normal", "collision", "deformed_vehicle"]
CHECKPOINT_PATH = "weights/image_classifier.pt"     # legacy EfficientNet weights
LOCAL_VIT_PATH  = "weights/image_classifier_vit"    # fine-tuned ViT (from training script)
HF_MODEL_ID     = "tiya1012/vit-accident-image"     # HuggingFace fallback


class ImageAccidentClassifier:
    """
    ViT-based accident image classifier.

    Uses tiya1012/vit-accident-image (F1=0.93) when available,
    falls back to EfficientNet-B1 (ImageNet pretrained) if offline.
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.model_type = None
        self.model_status = "unloaded"

        self._load_model()

    # ── Loading ──────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load in order: local fine-tuned ViT → HuggingFace ViT → EfficientNet fine-tuned → EfficientNet fallback."""

        # Option 1: Locally fine-tuned ViT (best — trained on your combined dataset)
        if os.path.isdir(LOCAL_VIT_PATH):
            try:
                self._load_local_vit()
                return
            except Exception as e:
                print(f"[WARNING] Local ViT weights failed: {e}")

        # Option 2: HuggingFace ViT (F1=0.93, auto-downloads ~330 MB)
        try:
            self._load_hf_vit()
            return
        except Exception as e:
            print(f"[WARNING] HuggingFace ViT failed: {e} — trying EfficientNet")

        # Option 3: Legacy EfficientNet fine-tuned weights
        if os.path.exists(CHECKPOINT_PATH):
            try:
                self._load_local_finetuned()
                return
            except Exception as e:
                print(f"[WARNING] Legacy checkpoint failed: {e}")

        # Option 4: EfficientNet ImageNet pretrained (always available)
        self._load_efficientnet_fallback()

    def _load_hf_vit(self):
        """Load ViT from local HuggingFace cache ONLY — never blocks on network."""
        from transformers import ViTForImageClassification, ViTImageProcessor

        print(f"[INFO] Trying ViT from local cache ({HF_MODEL_ID})...")
        # local_files_only=True raises immediately if not cached — no network call
        self.processor = ViTImageProcessor.from_pretrained(
            HF_MODEL_ID, local_files_only=True
        )
        self.model = ViTForImageClassification.from_pretrained(
            HF_MODEL_ID, local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_type   = "vit_hf"
        self.model_status = "vit_pretrained_hf"
        print(f"✅ ViT loaded from local cache (F1=0.93)")

    def _load_local_vit(self):
        """Load locally fine-tuned ViT from training/train_image_classifier.py output."""
        from transformers import ViTForImageClassification, ViTImageProcessor

        print(f"[INFO] Loading fine-tuned ViT from {LOCAL_VIT_PATH}...")
        self.processor = ViTImageProcessor.from_pretrained(LOCAL_VIT_PATH)
        self.model = ViTForImageClassification.from_pretrained(LOCAL_VIT_PATH)
        self.model.to(self.device)
        self.model.eval()
        self.model_type   = "vit_hf"
        self.model_status = "vit_fine_tuned_local"
        print(f"✅ Fine-tuned ViT loaded from {LOCAL_VIT_PATH} (highest accuracy)")

    def _load_local_finetuned(self):
        """Load user-fine-tuned EfficientNet weights from local checkpoint."""
        import timm
        from torchvision import transforms

        print(f"[INFO] Loading fine-tuned weights from {CHECKPOINT_PATH}...")
        self.model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=len(CLASSES))
        state = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "efficientnet_finetuned"
        self.model_status = "fine_tuned"
        self._build_efficientnet_transform()
        print(f"✅ Fine-tuned image classifier loaded from {CHECKPOINT_PATH}")

    def _load_efficientnet_fallback(self):
        """Load EfficientNet-B1 — tries cached ImageNet weights, skips download if offline."""
        import timm
        try:
            print("[INFO] Loading EfficientNet-B1 (ImageNet pretrained from timm cache)...")
            self.model = timm.create_model("efficientnet_b1", pretrained=True, num_classes=len(CLASSES))
            self.model_status = "efficientnet_imagenet"
            print("✅ EfficientNet-B1 loaded (ImageNet pretrained)")
        except Exception:
            print("[INFO] Offline — using EfficientNet-B1 without pretrained weights (physics scoring active)")
            self.model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=len(CLASSES))
            self.model_status = "efficientnet_no_pretrain"
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "efficientnet_imagenet"
        self._build_efficientnet_transform()

    def _build_efficientnet_transform(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ── Inference ────────────────────────────────────────────────────────────────

    def classify(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify a PIL image.

        Returns:
            {label, confidence, class_scores, model_status, is_accident}
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.model_type == "vit_hf":
            return self._classify_vit(image)
        else:
            return self._classify_efficientnet(image)

    def _classify_vit(self, image: Image.Image) -> Dict[str, Any]:
        """Run ViT inference using HuggingFace pipeline."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits  = outputs.logits                                  # (1, 2)
            probs   = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # id2label: {0: "Accident", 1: "No Accident"}
        pred_idx   = int(np.argmax(probs))
        vit_label  = VIT_ID2LABEL[pred_idx]
        app_label  = VIT_TO_APP_LABEL[vit_label]
        confidence = float(probs[pred_idx])

        # Build class_scores in our 3-class format
        acc_prob  = float(probs[0])   # "Accident"
        norm_prob = float(probs[1])   # "No Accident"

        # For deformed_vehicle: if strong accident signal but image is single vehicle,
        # the service layer will handle the routing — here we output the raw scores
        class_scores = {
            "normal":            round(norm_prob, 4),
            "collision":         round(acc_prob, 4),
            "deformed_vehicle":  round(acc_prob * 0.6, 4),   # soft proxy
        }

        return {
            "label":        app_label,
            "is_accident":  vit_label == "Accident",
            "confidence":   round(confidence, 4),
            "accident_probability": round(acc_prob, 4),
            "class_scores": class_scores,
            "model_status": self.model_status,
            "model_name":   HF_MODEL_ID,
        }

    def _classify_efficientnet(self, image: Image.Image) -> Dict[str, Any]:
        """Run EfficientNet inference (fallback)."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        label    = CLASSES[pred_idx]

        return {
            "label":        label,
            "is_accident":  label != "normal",
            "confidence":   round(float(probs[pred_idx]), 4),
            "accident_probability": round(float(probs[1]) + float(probs[2]), 4),
            "class_scores": {
                cls: round(float(probs[i]), 4)
                for i, cls in enumerate(CLASSES)
            },
            "model_status": self.model_status,
            "model_name":   "efficientnet_b1",
        }

    def classify_from_path(self, image_path: str) -> Dict[str, Any]:
        """Load image from file path and classify."""
        image = Image.open(image_path).convert("RGB")
        return self.classify(image)
