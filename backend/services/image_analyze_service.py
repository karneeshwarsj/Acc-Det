"""
Auto-routing image analysis service.

Automatically detects whether an image represents:
  - A 'complete' accident (vehicles overlapping/colliding OR visible deformation)
  - An 'incomplete' scenario (vehicles near each other, proximity risk table)

Detection logic:
  - Run YOLO first (fast)
  - If any vehicle pair has IoU > 0.05 → collision → complete classification
  - If single vehicle AND EfficientNet signals deformation → complete classification
  - If multiple vehicles with no overlap → incomplete → probability table
  - If no vehicles or single vehicle without deformation → complete classification (best effort)
"""

import os
import time
from typing import Dict, Any

from models.yolo_detector import YOLODetector
from services.image_service import classify_accident_image
from services.image_prob_service import estimate_proximity_probability

_detector: YOLODetector = None

OVERLAP_THRESHOLD = 0.05       # IoU above this → treat as collision in progress


def _get_detector():
    global _detector
    if _detector is None:
        _detector = YOLODetector(model_name="yolov8n.pt")
    return _detector


def _detect_image_type(image_path: str) -> Dict[str, Any]:
    """
    Quick scan of image to decide routing.

    Returns:
        {is_complete: bool, reason: str, vehicle_count: int, max_iou: float}
    """
    detector = _get_detector()

    try:
        result = detector.detect_vehicles_in_image(image_path)
    except Exception as e:
        return {"is_complete": True, "reason": f"detection_error: {e}", "vehicle_count": 0, "max_iou": 0.0}

    vehicle_count = result["vehicle_count"]
    max_iou       = result["max_iou"]
    overlapping   = result["vehicles_overlapping"]

    if vehicle_count == 0:
        return {
            "is_complete": True,
            "reason": "no_vehicles_detected → fallback to classification",
            "vehicle_count": 0,
            "max_iou": 0.0,
        }

    if overlapping or max_iou >= OVERLAP_THRESHOLD:
        return {
            "is_complete": True,
            "reason": f"vehicles_overlapping (IoU={max_iou:.3f})",
            "vehicle_count": vehicle_count,
            "max_iou": round(max_iou, 4),
        }

    if vehicle_count == 1:
        # Single vehicle — check for deformation via classification
        return {
            "is_complete": True,
            "reason": "single_vehicle → classify for deformation",
            "vehicle_count": 1,
            "max_iou": 0.0,
        }

    # Multiple vehicles, no overlap → proximity analysis
    return {
        "is_complete": False,
        "reason": f"multiple_vehicles_no_overlap (IoU={max_iou:.3f}) → proximity table",
        "vehicle_count": vehicle_count,
        "max_iou": round(max_iou, 4),
    }


def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Auto-detect image type and run the appropriate analysis pipeline.

    Returns merged result with extra 'analysis_mode' field:
        'complete_classification' → accident / deformation classification
        'incomplete_probability'  → proximity probability table
    """
    t_start = time.time()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1: Quick type detection
    detection = _detect_image_type(image_path)
    is_complete = detection["is_complete"]

    # Step 2: Route to correct pipeline
    if is_complete:
        result = classify_accident_image(image_path)
        result["analysis_mode"]  = "complete_classification"
        result["detection_info"] = detection
        result["mode_label"]     = "Accident Scene Analysis"
        result["mode_icon"]      = "🚗"
    else:
        result = estimate_proximity_probability(image_path)
        result["analysis_mode"]  = "incomplete_probability"
        result["detection_info"] = detection
        result["mode_label"]     = "Near-Miss Risk Analysis"
        result["mode_icon"]      = "📐"

    result["total_processing_time_seconds"] = round(time.time() - t_start, 2)
    return result
