"""Complete image accident classification service."""

import os
import time
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image

from models.yolo_detector import YOLODetector
from models.accident_classifier import ImageAccidentClassifier
from utils.frame_utils import frame_to_base64

_detector: YOLODetector = None
_classifier: ImageAccidentClassifier = None


def _get_models():
    global _detector, _classifier
    if _detector is None:
        _detector = YOLODetector()
    if _classifier is None:
        _classifier = ImageAccidentClassifier()
    return _detector, _classifier


def classify_accident_image(image_path: str) -> Dict[str, Any]:
    """
    Classify an uploaded image as an accident scene or normal.

    Strategy:
    - Run EfficientNet-B1 for holistic accident classification
    - Run YOLOv8 for vehicle detection and overlap analysis
    - Combine: if YOLO sees IoU > 0.05 → boost collision score
               if single heavily occluded/deformed bbox → boost deformed score

    Returns:
        {label, confidence, class_scores, vehicle_count, vehicles_overlapping,
         annotated_image_b64, model_status, processing_time_seconds}
    """
    t_start = time.time()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector, classifier = _get_models()

    # ── EfficientNet classification ────────────────────────────────────────────
    pil_image = Image.open(image_path).convert("RGB")
    clf_result = classifier.classify(pil_image)

    # ── YOLO vehicle detection ─────────────────────────────────────────────────
    yolo_result = detector.detect_vehicles_in_image(image_path)

    vehicle_count   = yolo_result["vehicle_count"]
    max_iou         = yolo_result["max_iou"]
    overlapping     = yolo_result["vehicles_overlapping"]

    # ── Fusion logic ───────────────────────────────────────────────────────────
    scores = dict(clf_result["class_scores"])  # {normal, collision, deformed_vehicle}

    if overlapping:
        # Vehicles clearly touching/overlapping → boost collision
        boost = min(0.3, max_iou * 0.6)
        scores["collision"] = min(0.99, scores["collision"] + boost)
        scores["normal"]    = max(0.01, scores["normal"] - boost)

    if vehicle_count == 1 and scores.get("deformed_vehicle", 0) > 0.25:
        # Single vehicle with strong deformation signal
        scores["deformed_vehicle"] = min(0.99, scores["deformed_vehicle"] + 0.1)
        scores["normal"]           = max(0.01, scores["normal"] - 0.1)

    if vehicle_count == 0:
        # No vehicles detected — fall back to pure model output
        pass

    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 4) for k, v in scores.items()}

    # Final label
    final_label = max(scores, key=scores.get)
    final_conf  = scores[final_label]

    # Map internal label to user-friendly label
    label_map = {
        "normal":           "No Accident",
        "collision":        "Accident — Collision",
        "deformed_vehicle": "Accident — Vehicle Deformation",
    }
    display_label = label_map.get(final_label, final_label)

    return {
        "label": display_label,
        "raw_label": final_label,
        "confidence": round(final_conf, 4),
        "class_scores": {
            label_map.get(k, k): round(v, 4) for k, v in scores.items()
        },
        "vehicle_count": vehicle_count,
        "vehicles_overlapping": overlapping,
        "max_vehicle_iou": max_iou,
        "annotated_image_b64": yolo_result["annotated_image_b64"],
        "model_status": clf_result["model_status"],
        "processing_time_seconds": round(time.time() - t_start, 2),
    }
