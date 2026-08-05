"""
Auto-routing video analysis service.

Automatically detects whether a video represents a 'complete' accident event
(collision already occurred) or an 'incomplete/ongoing' scenario (risk building up),
then routes to the correct pipeline.

Detection logic:
  - Sample early frames with YOLO
  - If any frame pair has IoU > 0.08 → vehicles overlapping → classify as complete accident
  - If peak risk score from TTC analysis > 0.80 → imminent collision → classify as complete
  - Otherwise → incomplete → return probability estimate
"""

import os
import time
from typing import Dict, Any

from models.yolo_detector import YOLODetector
from utils.frame_utils import extract_frames, compute_pairwise_features
from services.video_service import classify_complete_video
from services.video_prob_service import estimate_video_probability

_detector: YOLODetector = None

IOU_OVERLAP_THRESHOLD  = 0.15   # clear physical overlap of bounding boxes
RISK_COMPLETE_THRESHOLD = 0.85   # near-certain collision — only route here if extremely high


def _get_detector():
    global _detector
    if _detector is None:
        _detector = YOLODetector()
    return _detector


def _detect_video_type(video_path: str) -> Dict[str, Any]:
    """
    Quickly sample video frames to determine if an accident has already occurred.

    Returns:
        {is_complete: bool, reason: str, peak_iou: float, sample_frame_count: int}
    """
    detector = _get_detector()

    try:
        # Sample at 2 fps — just enough frames to detect overlaps
        frames, source_fps = extract_frames(video_path, target_fps=2)
    except Exception as e:
        # Default to incomplete on extraction failure
        return {"is_complete": False, "reason": f"extraction_fallback: {e}", "peak_iou": 0.0, "sample_frame_count": 0}

    if not frames:
        return {"is_complete": False, "reason": "no_frames", "peak_iou": 0.0, "sample_frame_count": 0}

    peak_iou   = 0.0
    peak_risk  = 0.0

    for frame in frames:
        detections = detector.detect_frame(frame)
        if len(detections) < 2:
            continue

        pairs = compute_pairwise_features(detections)
        for pair in pairs:
            iou  = pair["iou"]
            dist = pair["centroid_distance_px"]

            if iou > peak_iou:
                peak_iou = iou

            # Only count closing vehicles as risky
            # dist_risk: meaningful only when very close (< 80px)
            dist_risk = max(0.0, 1.0 - dist / 80.0)
            # iou must be substantial to contribute
            raw_risk = min(1.0, iou * 4.0 + dist_risk * 0.3)
            if raw_risk > peak_risk:
                peak_risk = raw_risk

    if peak_iou >= IOU_OVERLAP_THRESHOLD:
        return {
            "is_complete": True,
            "reason": f"vehicle_overlap_detected (peak IoU={peak_iou:.3f})",
            "peak_iou": round(peak_iou, 4),
            "peak_risk": round(peak_risk, 4),
            "sample_frame_count": len(frames),
        }

    if peak_risk >= RISK_COMPLETE_THRESHOLD:
        return {
            "is_complete": True,
            "reason": f"extreme_risk_score (peak risk={peak_risk:.3f})",
            "peak_iou": round(peak_iou, 4),
            "peak_risk": round(peak_risk, 4),
            "sample_frame_count": len(frames),
        }

    return {
        "is_complete": False,
        "reason": f"no_collision_detected (peak IoU={peak_iou:.3f}, peak risk={peak_risk:.3f})",
        "peak_iou": round(peak_iou, 4),
        "peak_risk": round(peak_risk, 4),
        "sample_frame_count": len(frames),
    }


def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Auto-detect video type and run the appropriate analysis pipeline.

    Returns merged result with an extra 'analysis_mode' field:
        'complete_classification' → accident event classification
        'incomplete_probability'  → risk probability estimate
    """
    t_start = time.time()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Step 1: Quick type detection
    detection = _detect_video_type(video_path)
    is_complete = detection["is_complete"]

    # Step 2: Route to correct pipeline
    if is_complete:
        result = classify_complete_video(video_path)
        result["analysis_mode"]  = "complete_classification"
        result["detection_info"] = detection
        result["mode_label"]     = "Complete Video — Accident Classification"
        result["mode_icon"]      = "🎬"
    else:
        result = estimate_video_probability(video_path)
        result["analysis_mode"]  = "incomplete_probability"
        result["detection_info"] = detection
        result["mode_label"]     = "Incomplete Video — Risk Probability"
        result["mode_icon"]      = "📊"

    result["total_processing_time_seconds"] = round(time.time() - t_start, 2)
    return result
