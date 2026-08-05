"""Near-miss image proximity probability service."""

import os
import time
from typing import Dict, Any

import cv2
import numpy as np

from models.yolo_detector import YOLODetector
from models.depth_estimator import DepthEstimator
from utils.physics import generate_probability_table
from utils.frame_utils import frame_to_base64

_detector: YOLODetector = None
_depth_estimator: DepthEstimator = None


def _get_models():
    global _detector, _depth_estimator
    if _detector is None:
        _detector = YOLODetector(model_name="yolov8n.pt")
    if _depth_estimator is None:
        _depth_estimator = DepthEstimator()
    return _detector, _depth_estimator


def estimate_proximity_probability(image_path: str) -> Dict[str, Any]:
    """
    For a near-miss image (vehicles close but not collided), estimate
    accident probability across multiple speed ranges.

    Pipeline:
      1. Detect vehicles with YOLOv8
      2. Find closest vehicle pair
      3. Estimate inter-vehicle distance (MiDaS + bbox heuristic)
      4. Generate probability table for each speed range

    Returns:
        {vehicle_count, estimated_distance_m, probability_table,
         annotated_image_b64, warning_message, processing_time_seconds}
    """
    t_start = time.time()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector, depth_estimator = _get_models()

    # ── Detect vehicles ────────────────────────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")

    detections = detector.detect_frame(frame)

    if len(detections) < 2:
        annotated = detector.annotate_frame(frame, detections)
        return {
            "vehicle_count": len(detections),
            "estimated_distance_m": None,
            "probability_table": [],
            "annotated_image_b64": frame_to_base64(annotated),
            "warning_message": (
                "Only one or no vehicles detected. "
                "Please upload an image with two or more vehicles for proximity analysis."
            ),
            "processing_time_seconds": round(time.time() - t_start, 2),
        }

    # ── Find closest vehicle pair ──────────────────────────────────────────────
    pair_distances = depth_estimator.estimate_all_pairs(frame, detections)

    if not pair_distances:
        return {
            "vehicle_count": len(detections),
            "estimated_distance_m": None,
            "probability_table": [],
            "annotated_image_b64": frame_to_base64(detector.annotate_frame(frame, detections)),
            "warning_message": "Could not estimate vehicle distances.",
            "processing_time_seconds": round(time.time() - t_start, 2),
        }

    closest_pair = min(pair_distances, key=lambda p: p["distance_m"])
    distance_m   = closest_pair["distance_m"]

    # ── Generate probability table ─────────────────────────────────────────────
    prob_table = generate_probability_table(distance_m)

    # ── Annotate image with distance overlay ──────────────────────────────────
    annotated = detector.annotate_frame(frame, detections)

    # Draw distance line between closest pair
    idx_a = closest_pair["vehicle_a"]
    idx_b = closest_pair["vehicle_b"]
    if idx_a < len(detections) and idx_b < len(detections):
        ca = detections[idx_a]["centroid"]
        cb = detections[idx_b]["centroid"]
        cv2.line(annotated, ca, cb, (255, 220, 0), 2)
        mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
        dist_label = f"~{distance_m:.1f}m"
        cv2.putText(
            annotated, dist_label,
            (mid[0] - 30, mid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 220, 0), 2, cv2.LINE_AA,
        )

    warning = None
    if distance_m < 2.0:
        warning = "Vehicles appear very close (< 2m). Even at low speeds, collision risk is very high."
    elif distance_m > 100.0:
        warning = "Estimated distance seems large. Distance estimation may be approximate for this image."

    return {
        "vehicle_count": len(detections),
        "estimated_distance_m": round(distance_m, 2),
        "all_pair_distances": [
            {k: v for k, v in p.items()} for p in pair_distances
        ],
        "closest_pair": closest_pair,
        "probability_table": prob_table,
        "annotated_image_b64": frame_to_base64(annotated),
        "warning_message": warning,
        "processing_time_seconds": round(time.time() - t_start, 2),
    }
