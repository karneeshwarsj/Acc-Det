"""Frame extraction and per-frame feature computation utilities."""

import base64
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np


# ─── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(video_path: str, target_fps: int = 5) -> Tuple[List[np.ndarray], float]:
    """
    Extract frames from a video at the target frame rate.

    Args:
        video_path: Path to video file
        target_fps: How many frames per second to sample

    Returns:
        (frames, source_fps) — list of BGR numpy arrays, original FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0

    frame_interval = max(1, int(source_fps / target_fps))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames, source_fps


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Return basic video metadata (duration, fps, resolution, frame count)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": round(fps, 2),
        "frame_count": int(frames),
        "duration_seconds": round(frames / fps, 2),
        "width": width,
        "height": height,
    }


# ─── Pairwise feature computation ───────────────────────────────────────────────

def compute_pairwise_features(detections: List[Dict[str, Any]]) -> List[Dict]:
    """
    For each pair of detected vehicles compute IoU and centroid distance.

    Args:
        detections: Output from YOLODetector.detect_frame()

    Returns:
        List of {pair_id, vehicle_a, vehicle_b, iou, centroid_distance_px}
    """
    pairs = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a = detections[i]
            b = detections[j]

            iou  = _iou(a["bbox"], b["bbox"])
            dist = _dist(a["centroid"], b["centroid"])

            pairs.append({
                "pair_id": f"{i}-{j}",
                "vehicle_a": i,
                "vehicle_b": j,
                "iou": round(iou, 4),
                "centroid_distance_px": round(dist, 2),
            })

    return pairs


def build_feature_vector(
    detections: List[Dict],
    pairs: List[Dict],
    velocity_map: Optional[Dict[int, float]] = None,
    trajectory_score: float = 0.0,
    frame_idx_norm: float = 0.0,
) -> np.ndarray:
    """
    Build a fixed-size 6D feature vector for a single frame:
    [vehicle_count, min_distance_norm, max_iou, avg_velocity_norm,
     trajectory_score, frame_idx_norm]

    Args:
        detections: Vehicle detections in this frame
        pairs: Pairwise features for this frame
        velocity_map: {vehicle_id: speed_px_per_frame}
        trajectory_score: Precomputed trajectory convergence score
        frame_idx_norm: Frame index normalized to [0, 1]

    Returns:
        numpy array of shape (6,)
    """
    vehicle_count   = min(len(detections), 10) / 10.0     # normalize to [0,1]

    if pairs:
        min_dist   = min(p["centroid_distance_px"] for p in pairs)
        max_iou    = max(p["iou"] for p in pairs)
        # Normalize distance by assuming 1500px as max expected distance
        min_dist_n = max(0.0, 1.0 - min_dist / 1500.0)
    else:
        min_dist_n = 0.0
        max_iou    = 0.0

    if velocity_map:
        velocities  = list(velocity_map.values())
        avg_vel     = sum(velocities) / len(velocities)
        avg_vel_n   = min(1.0, avg_vel / 50.0)  # normalize: 50 px/frame ≈ fast
    else:
        avg_vel_n = 0.0

    return np.array([
        vehicle_count,
        min_dist_n,
        max_iou,
        avg_vel_n,
        trajectory_score,
        frame_idx_norm,
    ], dtype=np.float32)


def build_sliding_windows(
    feature_vectors: List[np.ndarray],
    window_size: int = 16,
) -> List[np.ndarray]:
    """
    Create overlapping sliding windows from a sequence of feature vectors.

    Args:
        feature_vectors: List of (6,) arrays
        window_size: Number of frames per window

    Returns:
        List of (window_size, 6) arrays
    """
    windows = []
    n = len(feature_vectors)

    if n < window_size:
        # Pad with zeros at the start
        pad_count = window_size - n
        padded = [np.zeros(6, dtype=np.float32)] * pad_count + feature_vectors
        windows.append(np.stack(padded))
    else:
        step = max(1, window_size // 2)
        for start in range(0, n - window_size + 1, step):
            window = np.stack(feature_vectors[start : start + window_size])
            windows.append(window)

    return windows


# ─── Encoding helpers ────────────────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Encode an OpenCV BGR frame to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


# ─── Internal helpers ────────────────────────────────────────────────────────────

def _iou(bbox_a: List[int], bbox_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _dist(p1: tuple, p2: tuple) -> float:
    import math
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
