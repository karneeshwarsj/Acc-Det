"""Incomplete video accident probability estimation service."""

import os
import io
import time
import base64
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.yolo_detector import YOLODetector
from utils.frame_utils import extract_frames, get_video_metadata, frame_to_base64
from utils.tracking import VehicleTracker
from utils.physics import (
    compute_ttc,
    compute_risk_score,
    compute_trajectory_score,
    aggregate_probability,
)

_detector: YOLODetector = None

def _get_detector():
    global _detector
    if _detector is None:
        _detector = YOLODetector()
    return _detector


def estimate_video_probability(video_path: str) -> Dict[str, Any]:
    """
    Estimate accident probability from an incomplete/partial video clip.

    Pipeline:
      1. Extract all available frames
      2. Track vehicles across frames with ByteTrack
      3. Compute TTC, relative velocity, IoU for each vehicle pair per frame
      4. Aggregate risk across all frames and pairs
      5. Return probability score + risk heatmap

    Returns:
        {probability, risk_level, ttc_seconds, vehicle_pairs_analyzed,
         processing_time_seconds, risk_over_time, heatmap_b64, annotated_frames_b64}
    """
    t_start = time.time()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = _get_detector()
    tracker = VehicleTracker(max_history=30)

    try:
        metadata = get_video_metadata(video_path)
    except Exception:
        metadata = {}

    fps = metadata.get("fps", 5.0)
    sample_fps = min(5, fps)

    try:
        frames, source_fps = extract_frames(video_path, target_fps=int(sample_fps))
    except Exception as e:
        return {"error": f"Frame extraction failed: {str(e)}"}

    if not frames:
        return {"error": "No frames extracted from video."}

    # ── Per-frame tracking & risk computation ──────────────────────────────────
    per_frame_risks: List[float] = []
    all_pair_risks: List[Dict] = []
    annotated_frames_b64: List[str] = []
    total_pairs_seen = set()

    for frame_idx, frame in enumerate(frames):
        detections = detector.detect_frame(frame)

        if not detections:
            per_frame_risks.append(0.0)
            continue

        tracked = tracker.update(detections, frame)

        frame_pair_risks = []
        for i in range(len(tracked)):
            for j in range(i + 1, len(tracked)):
                a = tracked[i]
                b = tracked[j]
                tid_a = a.get("track_id", i)
                tid_b = b.get("track_id", j)
                pair_key = tuple(sorted([tid_a, tid_b]))
                total_pairs_seen.add(pair_key)

                # Centroid distance (pixels)
                cx_a, cy_a = a["centroid"]
                cx_b, cy_b = b["centroid"]
                dist_px = float(np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2))

                # Relative velocity (closing speed, px/frame)
                rel_vel = tracker.compute_relative_velocity(tid_a, tid_b, fps=sample_fps)
                vel_per_frame = rel_vel / sample_fps if sample_fps > 0 else 0.0

                # IoU
                ax1,ay1,ax2,ay2 = a["bbox"]
                bx1,by1,bx2,by2 = b["bbox"]
                ix1=max(ax1,bx1); iy1=max(ay1,by1)
                ix2=min(ax2,bx2); iy2=min(ay2,by2)
                inter=(max(0,ix2-ix1))*(max(0,iy2-iy1))
                union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
                iou = inter/union if union>0 else 0.0

                # TTC
                ttc = compute_ttc(dist_px, max(vel_per_frame, 0.0), fps=int(sample_fps))

                # Trajectory score
                hist_a = tracker.track_history.get(tid_a, [])
                hist_b = tracker.track_history.get(tid_b, [])
                traj = compute_trajectory_score(hist_a, hist_b)

                risk = compute_risk_score(ttc, iou, traj)

                frame_pair_risks.append({
                    "pair_id": f"{tid_a}-{tid_b}",
                    "probability": risk,
                    "ttc": ttc,
                    "iou": iou,
                    "trajectory_score": traj,
                    "distance_px": dist_px,
                })
                all_pair_risks.append({
                    "pair_id": f"{tid_a}-{tid_b}",
                    "probability": risk,
                    "ttc": ttc,
                })

        # Frame-level risk = max pair risk in this frame
        frame_risk = max((p["probability"] for p in frame_pair_risks), default=0.0)
        per_frame_risks.append(frame_risk)

        # Annotate every 4th frame (max 6 preview frames)
        if frame_idx % 4 == 0 and len(annotated_frames_b64) < 6:
            annotated = detector.annotate_frame(frame, detections)
            annotated_frames_b64.append(frame_to_base64(annotated))

    # ── Aggregate ──────────────────────────────────────────────────────────────
    if not all_pair_risks:
        return {
            "probability": 0.0,
            "risk_level": "Low",
            "ttc_seconds": 999.0,
            "vehicle_pairs_analyzed": 0,
            "message": "No vehicles detected in the video.",
            "processing_time_seconds": round(time.time() - t_start, 2),
            "risk_over_time": [],
            "heatmap_b64": None,
            "annotated_frames_b64": [],
        }

    agg = aggregate_probability(all_pair_risks)

    # ── Heatmap: risk over time ────────────────────────────────────────────────
    heatmap_b64 = _generate_risk_heatmap(per_frame_risks)

    return {
        "probability": agg["probability"],
        "risk_level": agg["risk_level"],
        "ttc_seconds": agg["min_ttc_seconds"],
        "vehicle_pairs_analyzed": len(total_pairs_seen),
        "frame_count": len(frames),
        "risk_over_time": [round(r, 4) for r in per_frame_risks],
        "heatmap_b64": heatmap_b64,
        "annotated_frames_b64": annotated_frames_b64,
        "video_metadata": metadata,
        "processing_time_seconds": round(time.time() - t_start, 2),
    }


def _generate_risk_heatmap(per_frame_risks: List[float]) -> str:
    """Generate a matplotlib risk-over-time chart encoded as base64 PNG."""
    try:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d0d1a")
        ax.set_facecolor("#0d0d1a")

        x = list(range(len(per_frame_risks)))
        y = per_frame_risks

        # Gradient fill
        ax.fill_between(x, y, alpha=0.35, color="#ef4444")
        ax.plot(x, y, color="#ef4444", linewidth=2.0, label="Risk Score")
        ax.axhline(0.5, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.7, label="High Risk")
        ax.axhline(0.75, color="#dc2626", linestyle="--", linewidth=1, alpha=0.7, label="Critical")

        ax.set_xlabel("Frame", color="#9ca3af", fontsize=9)
        ax.set_ylabel("Risk Score", color="#9ca3af", fontsize=9)
        ax.set_title("Accident Risk Over Time", color="#f3f4f6", fontsize=11, pad=10)
        ax.tick_params(colors="#6b7280")
        ax.spines["bottom"].set_color("#374151")
        ax.spines["left"].set_color("#374151")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#d1d5db", fontsize=8)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d0d1a")
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"[WARNING] Heatmap generation failed: {e}")
        return None
