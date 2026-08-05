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

IOU_OVERLAP_THRESHOLD  = 0.30   # stronger overlap threshold for a true collision event
RISK_COMPLETE_THRESHOLD = 0.96   # very high risk threshold for complete collision classification


def _get_detector():
    global _detector
    if _detector is None:
        _detector = YOLODetector(model_name="yolov8n.pt")  # use cached model
    return _detector


def _centroid_distance(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _match_detections(prev_detections: list, cur_detections: list, max_dist: float = 120.0) -> dict:
    """Match current detections to previous frame detections by nearest centroid."""
    mapping = {}
    used_prev = set()

    for cur_idx, cur_det in enumerate(cur_detections):
        best_prev = None
        best_dist = float("inf")
        for prev_idx, prev_det in enumerate(prev_detections):
            if prev_idx in used_prev:
                continue
            dist = _centroid_distance(cur_det["centroid"], prev_det["centroid"])
            if dist < best_dist:
                best_dist = dist
                best_prev = prev_idx

        if best_prev is not None and best_dist <= max_dist:
            mapping[cur_idx] = best_prev
            used_prev.add(best_prev)

    return mapping


def _z_from_bbox_area(area: float) -> float:
    return 1.0 / max(area ** 0.5, 1.0)


def _is_pair_closing(prev_a, prev_b, cur_a, cur_b, source_fps: float) -> bool:
    prev_dist = _centroid_distance(prev_a["centroid"], prev_b["centroid"])
    cur_dist = _centroid_distance(cur_a["centroid"], cur_b["centroid"])
    if prev_dist <= 0.0:
        return False

    distance_delta = prev_dist - cur_dist
    closing_rate = distance_delta * source_fps
    relative_shrink = distance_delta / prev_dist

    prev_z = _z_from_bbox_area(prev_a.get("area", 1.0) + prev_b.get("area", 1.0))
    cur_z = _z_from_bbox_area(cur_a.get("area", 1.0) + cur_b.get("area", 1.0))
    depth_delta = prev_z - cur_z

    return (
        distance_delta >= 10.0 or
        closing_rate >= 40.0 or
        relative_shrink >= 0.12 or
        depth_delta >= 0.03
    )


def _detect_video_type(video_path: str) -> Dict[str, Any]:
    """
    Quickly sample video frames to determine if an accident has already occurred.

    Returns:
        {is_complete: bool, reason: str, peak_iou: float, sample_frame_count: int}
    """
    detector = _get_detector()

    try:
        # Sample at 4 fps to catch brief collision moments in uploaded clips
        frames, source_fps = extract_frames(video_path, target_fps=4)
    except Exception as e:
        # Default to incomplete on extraction failure
        return {"is_complete": False, "reason": f"extraction_fallback: {e}", "peak_iou": 0.0, "sample_frame_count": 0}

    if not frames:
        return {"is_complete": False, "reason": "no_frames", "peak_iou": 0.0, "sample_frame_count": 0}

    peak_iou   = 0.0
    peak_risk  = 0.0
    overlap_closing_detected = False
    prev_detections = None

    for frame in frames:
        detections = detector.detect_frame(frame)
        if len(detections) < 2:
            prev_detections = detections
            continue

        pairs = compute_pairwise_features(detections)
        mapping = _match_detections(prev_detections, detections) if prev_detections else {}

        for pair in pairs:
            iou  = pair["iou"]
            dist = pair["centroid_distance_px"]

            if iou > peak_iou:
                peak_iou = iou

            # Only count closing vehicles as risky
            # dist_risk: meaningful only when very close (< 80px)
            dist_risk = max(0.0, 1.0 - dist / 80.0)
            # Larger IoU and very tight proximity should dominate collision risk
            raw_risk = min(1.0, iou * 5.0 + dist_risk * 0.2)
            if raw_risk > peak_risk:
                peak_risk = raw_risk

            if iou >= IOU_OVERLAP_THRESHOLD and prev_detections:
                a_idx = pair["vehicle_a"]
                b_idx = pair["vehicle_b"]
                if a_idx in mapping and b_idx in mapping:
                    prev_a = prev_detections[mapping[a_idx]]
                    prev_b = prev_detections[mapping[b_idx]]
                    if _is_pair_closing(prev_a, prev_b, detections[a_idx], detections[b_idx], source_fps):
                        overlap_closing_detected = True

        prev_detections = detections

    if overlap_closing_detected:
        return {
            "is_complete": True,
            "reason": f"vehicle_overlap_closing_detected (peak IoU={peak_iou:.3f})",
            "peak_iou": round(peak_iou, 4),
            "peak_risk": round(peak_risk, 4),
            "sample_frame_count": len(frames),
        }

    if peak_risk >= RISK_COMPLETE_THRESHOLD:
        return {
            "is_complete": True,
            "reason": f"high_risk_detected (peak risk={peak_risk:.3f})",
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
        # Map scalar probability into class-wise distribution for frontend
        # Ensure 'probability' exists (aggregate_probability -> probability)
        prob = result.get("probability")
        if prob is not None:
            # probability is in [0,1]
            result["class_scores"] = {
                "Accident": round(float(prob), 4),
                "No Accident": round(max(0.0, 1.0 - float(prob)), 4),
            }
            # Add a label for consistency with classification responses
            result["label"] = "Accident" if prob >= 0.5 else "No Accident"
            result["confidence"] = round(float(prob) if prob >= 0.5 else 1.0 - float(prob), 4)

        result["analysis_mode"]  = "incomplete_probability"
        result["detection_info"] = detection
        result["mode_label"]     = "Incomplete Video — Risk Probability"
        result["mode_icon"]      = "📊"

    result["total_processing_time_seconds"] = round(time.time() - t_start, 2)
    return result
