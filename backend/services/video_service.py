"""Complete video accident classification service."""

import os
import time
from typing import Dict, Any, List

import numpy as np

from models.yolo_detector import YOLODetector
from models.video_classifier import VideoAccidentClassifier
from utils.frame_utils import (
    extract_frames,
    get_video_metadata,
    compute_pairwise_features,
    build_feature_vector,
    build_sliding_windows,
    frame_to_base64,
)
from utils.physics import compute_trajectory_score, compute_ttc, compute_risk_score, aggregate_probability


# Collision override threshold for videos with visible overlapping vehicles
VIDEO_COLLISION_IOU_THRESHOLD = 0.20


def _centroid_distance(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _is_collision_override(prev_centroids: Dict[int, tuple], detections: List[Dict[str, Any]], overlap_candidates: List[Dict[str, Any]]) -> bool:
    if not prev_centroids:
        return False

    for pair in overlap_candidates:
        a_idx = pair["vehicle_a"]
        b_idx = pair["vehicle_b"]
        if a_idx not in prev_centroids or b_idx not in prev_centroids:
            continue

        prev_a = {"centroid": prev_centroids[a_idx], "area": 1.0}
        prev_b = {"centroid": prev_centroids[b_idx], "area": 1.0}
        cur_a = detections[a_idx]
        cur_b = detections[b_idx]

        prev_dist = _centroid_distance(prev_a["centroid"], prev_b["centroid"])
        cur_dist = _centroid_distance(cur_a["centroid"], cur_b["centroid"])
        if prev_dist <= 0:
            continue

        distance_delta = prev_dist - cur_dist
        if distance_delta >= 14.0:
            return True

        if distance_delta / prev_dist >= 0.12:
            return True

    return False

# Module-level singletons (initialized on first use)
_detector: YOLODetector = None
_classifier: VideoAccidentClassifier = None


def _get_models():
    global _detector, _classifier
    if _detector is None:
        _detector = YOLODetector(model_name="yolov8n.pt")
    if _classifier is None:
        _classifier = VideoAccidentClassifier()
    return _detector, _classifier


def classify_complete_video(video_path: str) -> Dict[str, Any]:
    """
    Classify a complete video clip as Accident or No Accident.

    Pipeline:
      1. Extract frames at 5 fps
      2. Detect vehicles in each frame (YOLOv8)
      3. Build 6D feature vectors per frame
      4. Run LSTM classifier on sliding windows of 16 frames
      5. Aggregate results (majority vote + max confidence)

    Returns:
        {label, confidence, class_scores, frame_count, processing_time_seconds,
         annotated_frames_b64, video_metadata, model_status}
    """
    t_start = time.time()

    # ── Validate file ──────────────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector, classifier = _get_models()

    # ── Metadata ───────────────────────────────────────────────────────────────
    try:
        metadata = get_video_metadata(video_path)
    except Exception:
        metadata = {}

    duration = metadata.get("duration_seconds", 0)
    if duration > 0 and duration < 1.0:
        return {
            "error": "Video too short (< 1 second). Please upload at least 1 second of footage.",
            "processing_time_seconds": round(time.time() - t_start, 2),
        }

    # ── Extract frames ─────────────────────────────────────────────────────────
    try:
        frames, source_fps = extract_frames(video_path, target_fps=5)
    except Exception as e:
        return {"error": f"Frame extraction failed: {str(e)}"}

    if len(frames) == 0:
        return {"error": "No frames could be extracted from the video."}

    # ── Per-frame processing ───────────────────────────────────────────────────
    feature_vectors: List[np.ndarray] = []
    annotated_frames_b64: List[str] = []
    total_vehicles_detected = 0
    prev_centroids: Dict[int, tuple] = {}
    collision_override = False

    for idx, frame in enumerate(frames):
        detections = detector.detect_frame(frame)
        total_vehicles_detected += len(detections)
        pairs = compute_pairwise_features(detections)

        # Only consider collision override when overlap is strong and motion is closing
        overlap_candidates = [pair for pair in pairs if pair["iou"] >= VIDEO_COLLISION_IOU_THRESHOLD]
        if overlap_candidates and len(detections) >= 2:
            if _is_collision_override(prev_centroids, detections, overlap_candidates):
                collision_override = True

        # Build velocity map from centroid displacement
        velocity_map: Dict[int, float] = {}
        for det in detections:
            vid = det["id"]
            if vid in prev_centroids:
                dx = det["centroid"][0] - prev_centroids[vid][0]
                dy = det["centroid"][1] - prev_centroids[vid][1]
                velocity_map[vid] = float(np.sqrt(dx**2 + dy**2))
            prev_centroids[vid] = det["centroid"]

        # Trajectory score (simplified: use velocity difference as proxy)
        traj_score = 0.0
        if len(detections) >= 2:
            histories = [[d["centroid"]] for d in detections]
            if len(histories) >= 2:
                # Placeholder — full trajectory needs history; starts meaningful after frame 2
                traj_score = min(1.0, len(pairs) * 0.1) if pairs else 0.0

        fv = build_feature_vector(
            detections=detections,
            pairs=pairs,
            velocity_map=velocity_map,
            trajectory_score=traj_score,
            frame_idx_norm=idx / max(len(frames) - 1, 1),
        )
        feature_vectors.append(fv)

        # Annotate every 5th frame (for preview, max 8 frames)
        if idx % 5 == 0 and len(annotated_frames_b64) < 8:
            annotated = detector.annotate_frame(frame, detections)
            annotated_frames_b64.append(frame_to_base64(annotated))

    # ── Collision override ─────────────────────────────────────────────────
    if collision_override:
        t_elapsed = round(time.time() - t_start, 2)
        return {
            "label": "Accident",
            "confidence": 1.0,
            "class_scores": {"Accident": 1.0, "No Accident": 0.0},
            "frame_count": len(frames),
            "vehicle_detections_total": total_vehicles_detected,
            "windows_analyzed": len(frames),
            "annotated_frames_b64": annotated_frames_b64,
            "video_metadata": metadata,
            "processing_time_seconds": t_elapsed,
            "model_status": "collision_override",
        }

    # ── ViT frame-level scoring (primary — high accuracy) ─────────────────────
    # Use score_frames() when the classifier has the ViT loaded.
    # Fall back to feature-vector path + physics if ViT is unavailable.
    if hasattr(classifier, 'score_frames') and classifier._get_classifier() is not None:
        result = classifier.score_frames(frames)
    else:
        # Physics-based fallback from feature vectors
        windows = build_sliding_windows(feature_vectors, window_size=16)
        result  = classifier.predict_windows(windows)

        if result["model_status"] in ("random_init", "vit_frame_aggregation") and feature_vectors:
            all_pair_risks = []
            for fv in feature_vectors:
                iou           = float(np.clip(fv[2], 0.0, 1.0))
                dist_norm     = float(np.clip(fv[1], 0.0, 1.0))
                vel_norm      = float(np.clip(fv[3], 0.0, 1.0))
                traj          = float(np.clip(fv[4], 0.0, 1.0))
                dist_px       = (1.0 - dist_norm) * 1500.0
                vel_px        = vel_norm * 50.0
                ttc_est       = max(0.5, dist_px / (vel_px * 5.0)) if vel_px > 0.5 else 999.0
                risk          = compute_risk_score(ttc_est, iou, traj)
                all_pair_risks.append({"probability": risk, "ttc": ttc_est, "pair_id": "physics"})

            agg        = aggregate_probability(all_pair_risks)
            phys_prob  = agg["probability"]
            phys_label = "Accident" if phys_prob >= 0.55 else "No Accident"
            result = {
                "label":        phys_label,
                "confidence":   round(phys_prob if phys_label == "Accident" else 1.0 - phys_prob, 4),
                "class_scores": {
                    "Accident":    round(phys_prob, 4),
                    "No Accident": round(1.0 - phys_prob, 4),
                },
                "window_count": len(windows),
                "model_status": "physics_fallback",
            }

    t_elapsed = round(time.time() - t_start, 2)

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "class_scores": result["class_scores"],
        "frame_count": len(frames),
        "vehicle_detections_total": total_vehicles_detected,
        "windows_analyzed": result.get("window_count", len(frames)),
        "annotated_frames_b64": annotated_frames_b64,
        "video_metadata": metadata,
        "processing_time_seconds": t_elapsed,
        "model_status": result["model_status"],
    }
