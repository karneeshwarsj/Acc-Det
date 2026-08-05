"""Vehicle tracking using supervision's ByteTrack."""

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np


class VehicleTracker:
    """
    Lightweight vehicle tracker using supervision's ByteTrack.
    Maintains per-track centroid history for velocity estimation.
    """

    def __init__(self, max_history: int = 30):
        """
        Args:
            max_history: Maximum number of past positions to keep per track.
        """
        self.max_history = max_history
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._tracker = None
        self._init_tracker()

    def _init_tracker(self):
        """Initialize ByteTrack tracker from supervision."""
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack()
        except Exception as e:
            print(f"[WARNING] ByteTrack init failed: {e}. Using fallback ID tracker.")
            self._tracker = None

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.

        Args:
            detections: Output of YOLODetector.detect_frame()
            frame: Current video frame (unused in this impl, kept for API compat)

        Returns:
            List of tracked vehicles with stable track_id assigned.
        """
        if not detections:
            return []

        if self._tracker is not None:
            return self._update_with_bytetrack(detections)
        else:
            return self._update_fallback(detections)

    def _update_with_bytetrack(self, detections: List[Dict]) -> List[Dict]:
        """Use supervision ByteTrack."""
        import supervision as sv

        xyxy   = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confs  = np.array([d["confidence"] for d in detections], dtype=np.float32)
        cls_ids = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=confs, class_id=cls_ids)
        tracked = self._tracker.update_with_detections(sv_dets)

        result = []
        for i, tid in enumerate(tracked.tracker_id):
            if i >= len(detections):
                break
            det = detections[i].copy()
            det["track_id"] = int(tid)
            cx, cy = det["centroid"]
            history = self.track_history[int(tid)]
            history.append((cx, cy))
            if len(history) > self.max_history:
                history.pop(0)
            det["history"] = list(history)
            result.append(det)

        return result

    def _update_fallback(self, detections: List[Dict]) -> List[Dict]:
        """Simple fallback: assign sequential IDs per frame."""
        result = []
        for i, det in enumerate(detections):
            det = det.copy()
            det["track_id"] = i
            cx, cy = det["centroid"]
            self.track_history[i].append((cx, cy))
            if len(self.track_history[i]) > self.max_history:
                self.track_history[i].pop(0)
            det["history"] = list(self.track_history[i])
            result.append(det)
        return result

    def get_velocity(self, track_id: int, fps: float = 5.0) -> float:
        """
        Estimate speed of a track in pixels/second.

        Args:
            track_id: Track ID
            fps: Frames per second (frames between history entries)

        Returns:
            Speed in pixels/second
        """
        history = self.track_history.get(track_id, [])
        if len(history) < 2:
            return 0.0

        # Use last two known positions
        p1 = history[-2]
        p2 = history[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = np.sqrt(dx**2 + dy**2)
        return dist * fps  # px/second

    def get_velocity_per_frame(self, track_id: int) -> float:
        """Speed in pixels per frame (unscaled)."""
        history = self.track_history.get(track_id, [])
        if len(history) < 2:
            return 0.0
        p1 = history[-2]
        p2 = history[-1]
        return float(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))

    def compute_relative_velocity(
        self,
        track_id_a: int,
        track_id_b: int,
        fps: float = 5.0,
    ) -> float:
        """
        Estimate the approach (closing) speed between two tracks in px/second.
        Positive = closing, negative = separating.
        """
        hist_a = self.track_history.get(track_id_a, [])
        hist_b = self.track_history.get(track_id_b, [])

        if len(hist_a) < 2 or len(hist_b) < 2:
            return 0.0

        dist_prev = np.sqrt(
            (hist_a[-2][0] - hist_b[-2][0])**2 +
            (hist_a[-2][1] - hist_b[-2][1])**2
        )
        dist_curr = np.sqrt(
            (hist_a[-1][0] - hist_b[-1][0])**2 +
            (hist_a[-1][1] - hist_b[-1][1])**2
        )

        # Positive = distance is decreasing = closing
        delta = (dist_prev - dist_curr) * fps
        return float(delta)

    def reset(self):
        """Clear all track history."""
        self.track_history.clear()
        self._init_tracker()
