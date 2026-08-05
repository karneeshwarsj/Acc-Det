"""
ViT-based video accident classifier.

Instead of a random-init LSTM, this uses the same fine-tuned ViT
(tiya1012/vit-accident-image, F1=0.93) to score every sampled frame,
then aggregates scores across the sequence:
  - peak_score:   worst-case frame (catches accident moments)
  - mean_score:   overall scene risk level
  - final:        0.7 * peak + 0.3 * mean

This approach:
  - Requires NO separate video training
  - Immediately benefits from the ViT's image-level accuracy
  - Correctly separates "one dangerous moment" from "ongoing risk"
"""

from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image


class VideoAccidentClassifier:
    """
    Frame-level ViT scoring for video accident classification.
    Reuses the same ViT model as the image classifier.
    """

    LABELS = ["No Accident", "Accident"]

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._classifier = None   # lazy-loaded to avoid double model load
        self.model_status = "vit_frame_aggregation"

    def _get_classifier(self):
        """Lazy-load the shared ImageAccidentClassifier."""
        if self._classifier is None:
            from models.accident_classifier import ImageAccidentClassifier
            self._classifier = ImageAccidentClassifier(device=self.device)
            self.model_status = self._classifier.model_status
        return self._classifier

    # ── Frame-level scoring ───────────────────────────────────────────────────────

    def score_frame(self, frame_bgr: np.ndarray) -> float:
        """Score a single BGR frame and return accident probability [0..1]."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        result = self._get_classifier().classify(pil)
        return float(result.get("accident_probability", 0.5))

    def score_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Score a list of BGR frames using the ViT and aggregate.
        HIGH-ACCURACY PATH — call this when raw frames are available.

        Returns:
            {label, confidence, class_scores, per_frame_scores,
             peak_score, mean_score, model_status}
        """
        if not frames:
            return self._empty_result()

        per_frame_scores = [self.score_frame(f) for f in frames]

        peak  = max(per_frame_scores)
        mean  = sum(per_frame_scores) / len(per_frame_scores)
        # Weighted blend: peak captures accidents, mean prevents single noisy frames
        final = min(0.99, 0.70 * peak + 0.30 * mean)

        label = "Accident" if final >= 0.50 else "No Accident"
        return {
            "label":             label,
            "confidence":        round(final if label == "Accident" else 1.0 - final, 4),
            "class_scores": {
                "Accident":    round(final, 4),
                "No Accident": round(1.0 - final, 4),
            },
            "per_frame_scores":  [round(s, 4) for s in per_frame_scores],
            "peak_score":        round(peak, 4),
            "mean_score":        round(mean, 4),
            "window_count":      len(frames),
            "model_status":      self.model_status,
        }

    # ── Feature-vector path (fallback, used by video_service physics override) ──

    def predict_windows(self, windows: List[np.ndarray]) -> Dict[str, Any]:
        """
        Fallback: derive risk from 6D physics feature vectors when frames
        are unavailable. video_service.py's physics override takes precedence
        over this output when model_status != 'fine_tuned'.

        Feature layout: [vehicle_count_norm, min_dist_norm, max_iou,
                          avg_vel_norm, traj, frame_norm]
        """
        if not windows:
            return self._empty_result()

        window_scores = []
        for w in windows:
            max_iou  = float(np.max(w[:, 2]))    # peak IoU across window
            avg_vel  = float(np.mean(w[:, 3]))   # avg velocity
            min_dist = float(np.max(w[:, 1]))    # highest closeness (1 = very close)
            traj     = float(np.mean(w[:, 4]))   # avg trajectory convergence

            score = min(1.0,
                max_iou  * 4.0 +
                avg_vel  * 0.3 +
                min_dist * 0.2 +
                traj     * 0.3
            )
            window_scores.append(score)

        peak  = max(window_scores)
        mean  = sum(window_scores) / len(window_scores)
        final = min(0.99, 0.70 * peak + 0.30 * mean)

        label = "Accident" if final >= 0.50 else "No Accident"
        return {
            "label":        label,
            "confidence":   round(final if label == "Accident" else 1.0 - final, 4),
            "class_scores": {
                "Accident":    round(final, 4),
                "No Accident": round(1.0 - final, 4),
            },
            "window_count": len(windows),
            "model_status": self.model_status,
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "label": "No Accident",
            "confidence": 0.5,
            "class_scores": {"No Accident": 0.5, "Accident": 0.5},
            "model_status": self.model_status,
        }
