"""
AccidentAI — Dataset Preparation Helper

This script provides:
  1. Dataset download links and instructions
  2. Directory structure setup
  3. Video-to-feature extraction pipeline

Usage:
  python dataset_prep.py --setup          # Create expected directory structure
  python dataset_prep.py --extract-video  # Extract features from video clips
"""

import os
import sys
import argparse
import glob

import numpy as np

# Add backend to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


DATASET_LINKS = """
╔══════════════════════════════════════════════════════════════════════╗
║                   RECOMMENDED DATASETS                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📹 VIDEO DATASETS (for Phase 1):                                   ║
║                                                                      ║
║  1. CADP (CCTV Accident Dataset for Prediction)                     ║
║     ~1,416 clips · Traffic surveillance footage                     ║
║     Link: https://ankitshah009.github.io/accident_forecasting       ║
║                                                                      ║
║  2. Kaggle: Car Accident Detection from CCTV                       ║
║     ~630 clips · Labeled accident/non-accident                      ║
║     Link: https://www.kaggle.com/datasets/ckay16/                   ║
║           accident-detection-from-cctv-footage                       ║
║                                                                      ║
║  3. UCF-Crime Dataset (Car Accident subset)                         ║
║     Link: https://www.crcv.ucf.edu/projects/real-world/             ║
║                                                                      ║
║  🖼️  IMAGE DATASETS (for Phase 2):                                  ║
║                                                                      ║
║  1. CarDD (Car Damage Detection Dataset)                            ║
║     ~4,000 images · 6 damage categories                             ║
║     Link: https://github.com/CarDD-USTC/CarDD-Release              ║
║                                                                      ║
║  2. Kaggle: Car Damage Severity Classification                     ║
║     5 severity levels · ~3,000 images                               ║
║     Search: "Car Damage Detection Dataset" on Kaggle                ║
║                                                                      ║
║  3. Roboflow Universe: Vehicle Damage Detection                     ║
║     YOLO-format annotations · ~1,200 images                         ║
║     Link: https://universe.roboflow.com (search "vehicle damage")   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
1. Download datasets from the links above
2. Place video clips in:
     data/videos/accident/     and  data/videos/no_accident/
3. Place images in:
     data/images/train/{normal,collision,deformed_vehicle}/
     data/images/val/{normal,collision,deformed_vehicle}/
4. Run:  python dataset_prep.py --extract-video
   to generate feature .npy files for LSTM training
"""


def setup_directories():
    """Create the expected directory structure for datasets."""
    dirs = [
        "data/videos/accident",
        "data/videos/no_accident",
        "data/video_features/accident",
        "data/video_features/no_accident",
        "data/images/train/normal",
        "data/images/train/collision",
        "data/images/train/deformed_vehicle",
        "data/images/val/normal",
        "data/images/val/collision",
        "data/images/val/deformed_vehicle",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  📁 {d}/")

    print("\n✅ Directory structure created!")
    print(DATASET_LINKS)


def extract_video_features():
    """
    Extract per-frame feature vectors from video clips.
    Saves as .npy files for LSTM training.
    """
    try:
        from models.yolo_detector import YOLODetector
        from utils.frame_utils import extract_frames, compute_pairwise_features, build_feature_vector
    except ImportError:
        print("❌ Cannot import backend modules. Run from the training/ directory.")
        print("   Or ensure ../backend is accessible.")
        return

    detector = YOLODetector()

    # Try to load ViT image classifier for per-frame logits/embeddings
    vit_classifier = None
    try:
        from models.accident_classifier import ImageAccidentClassifier
        vit_classifier = ImageAccidentClassifier()
        print(f"[INFO] ViT classifier available ({vit_classifier.model_status}), will extract logits+embeddings where possible.")
    except Exception:
        print("[WARN] ViT classifier not available; saving only the 6-D physics features.")
    processed = 0

    for label in ["accident", "no_accident"]:
        video_dir  = f"data/videos/{label}"
        output_dir = f"data/video_features/{label}"
        os.makedirs(output_dir, exist_ok=True)

        video_files = (
            glob.glob(os.path.join(video_dir, "*.mp4")) +
            glob.glob(os.path.join(video_dir, "*.avi")) +
            glob.glob(os.path.join(video_dir, "*.mov"))
        )

        print(f"\n🎬 Processing {len(video_files)} {label} videos...")

        for vpath in video_files:
            try:
                frames, fps = extract_frames(vpath, target_fps=5)
                feature_vectors = []
                vit_logits_list = []
                vit_emb_list = []

                prev_centroids = {}
                for idx, frame in enumerate(frames):
                    detections = detector.detect_frame(frame)
                    pairs = compute_pairwise_features(detections)

                    velocity_map = {}
                    for det in detections:
                        vid = det["id"]
                        if vid in prev_centroids:
                            dx = det["centroid"][0] - prev_centroids[vid][0]
                            dy = det["centroid"][1] - prev_centroids[vid][1]
                            velocity_map[vid] = float(np.sqrt(dx**2 + dy**2))
                        prev_centroids[vid] = det["centroid"]

                    fv = build_feature_vector(
                        detections=detections,
                        pairs=pairs,
                        velocity_map=velocity_map,
                        trajectory_score=0.0,
                        frame_idx_norm=idx / max(len(frames)-1, 1),
                    )
                    feature_vectors.append(fv)

                    # ViT per-frame features (logits + embedding)
                    try:
                        if vit_classifier is not None and vit_classifier.model_type == "vit_hf":
                            # Convert BGR (cv2) -> RGB PIL Image
                            from PIL import Image
                            import cv2 as _cv

                            pil = Image.fromarray(_cv.cvtColor(frame, _cv.COLOR_BGR2RGB)).convert("RGB")
                            inputs = vit_classifier.processor(images=pil, return_tensors="pt").to(vit_classifier.device)
                            import torch as _torch
                            with _torch.no_grad():
                                out = vit_classifier.model(**inputs, output_hidden_states=True)
                                logits = out.logits.squeeze(0).cpu().numpy()       # (2,)
                                # last hidden layer, CLS token at index 0
                                if hasattr(out, "hidden_states") and out.hidden_states is not None:
                                    last = out.hidden_states[-1].squeeze(0)
                                    emb = last[0].cpu().numpy()
                                else:
                                    # fallback: mean-pool last hidden state if available
                                    try:
                                        last = out.last_hidden_state.squeeze(0)
                                        emb = last.mean(0).cpu().numpy()
                                    except Exception:
                                        emb = np.zeros(768, dtype=np.float32)

                            vit_logits_list.append(logits.astype(np.float32))
                            vit_emb_list.append(emb.astype(np.float32))
                        else:
                            vit_logits_list.append(np.zeros(2, dtype=np.float32))
                            vit_emb_list.append(np.zeros(768, dtype=np.float32))
                    except Exception:
                        vit_logits_list.append(np.zeros(2, dtype=np.float32))
                        vit_emb_list.append(np.zeros(768, dtype=np.float32))

                if feature_vectors:
                    arr = np.stack(feature_vectors)
                    basename = os.path.splitext(os.path.basename(vpath))[0]
                    out_base = os.path.join(output_dir, f"{basename}")

                    # Save base 6-D features (backwards compatibility)
                    np.save(out_base + ".npy", arr)

                    # Save extended per-frame ViT features alongside physics features
                    vit_logits_arr = np.stack(vit_logits_list) if vit_logits_list else np.zeros((arr.shape[0], 2), dtype=np.float32)
                    vit_emb_arr    = np.stack(vit_emb_list)    if vit_emb_list else np.zeros((arr.shape[0], 768), dtype=np.float32)
                    np.savez_compressed(out_base + ".npz", features=arr, vit_logits=vit_logits_arr, vit_emb=vit_emb_arr)

                    processed += 1
                    print(f"   ✅ {basename} → features:{arr.shape} vit_logits:{vit_logits_arr.shape} vit_emb:{vit_emb_arr.shape}")

            except Exception as e:
                print(f"   ⚠️  Skipped {os.path.basename(vpath)}: {e}")

    print(f"\n🎉 Feature extraction complete! {processed} videos processed.")
    print(f"   Output: data/video_features/{{accident,no_accident}}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation for AccidentAI")
    parser.add_argument("--setup", action="store_true", help="Create directory structure")
    parser.add_argument("--extract-video", action="store_true", help="Extract features from video clips")
    args = parser.parse_args()

    if args.setup:
        setup_directories()
    elif args.extract_video:
        extract_video_features()
    else:
        print(DATASET_LINKS)
        print("\nRun with --setup to create directories, or --extract-video to process clips.")
