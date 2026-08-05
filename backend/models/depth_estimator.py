"""MiDaS monocular depth estimator for inter-vehicle distance estimation."""

import warnings
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
import torch


# Real-world reference: average car width ≈ 1.8m used for pixel-to-meter calibration
AVG_VEHICLE_WIDTH_M = 1.8
PIXELS_PER_METER_FALLBACK = 80  # Rough fallback if calibration fails


class DepthEstimator:
    """
    Wraps Intel MiDaS (MiDaS_small) for monocular relative depth estimation.
    Provides inter-vehicle distance estimation in approximate meters.
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._transform = None
        self._load_model()

    def _load_model(self):
        """Load MiDaS model from torch hub (downloads once, then cached)."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = torch.hub.load(
                    "intel-isl/MiDaS",
                    "MiDaS_small",
                    verbose=False,
                )
                transforms_hub = torch.hub.load(
                    "intel-isl/MiDaS",
                    "transforms",
                    verbose=False,
                )
                self._transform = transforms_hub.small_transform

            self._model.to(self.device)
            self._model.eval()
            print("✅ MiDaS depth estimator loaded.")
        except Exception as e:
            print(f"[WARNING] MiDaS failed to load: {e}")
            print("[INFO] Falling back to bounding-box-size heuristic for distance estimation.")
            self._model = None

    def get_depth_map(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute MiDaS depth map for an image.

        Args:
            image_bgr: OpenCV BGR numpy array

        Returns:
            Normalized depth map (0=far, 1=close) or None if model unavailable
        """
        if self._model is None:
            return None

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self._transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to [0, 1]  (MiDaS outputs inverse depth: higher = closer)
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)

        return depth_map

    def estimate_distance_between_vehicles(
        self,
        image_bgr: np.ndarray,
        bbox_a: List[int],
        bbox_b: List[int],
        detections: Optional[List[Dict]] = None,
    ) -> float:
        """
        Estimate real-world distance between two vehicles in meters.

        Strategy:
        1. Use MiDaS depth values at vehicle centroids for depth difference
        2. Combine with pixel distance and bounding box size heuristic for scaling
        3. Fall back to pure bounding-box heuristic if MiDaS unavailable

        Args:
            image_bgr: Full image (BGR OpenCV array)
            bbox_a: [x1, y1, x2, y2] of vehicle A
            bbox_b: [x1, y1, x2, y2] of vehicle B
            detections: Optional full detection list (used for calibration)

        Returns:
            Estimated distance in meters (approximate)
        """
        # Centroids
        cx_a = (bbox_a[0] + bbox_a[2]) // 2
        cy_a = (bbox_a[1] + bbox_a[3]) // 2
        cx_b = (bbox_b[0] + bbox_b[2]) // 2
        cy_b = (bbox_b[1] + bbox_b[3]) // 2

        # Pixel distance between centroids
        pixel_dist = float(np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2))

        # Estimate pixels-per-meter from bounding box widths
        width_a = bbox_a[2] - bbox_a[0]
        width_b = bbox_b[2] - bbox_b[0]
        avg_width_px = (width_a + width_b) / 2.0

        if avg_width_px > 10:
            px_per_m = avg_width_px / AVG_VEHICLE_WIDTH_M
        else:
            px_per_m = PIXELS_PER_METER_FALLBACK

        if self._model is not None:
            # Adjust using depth difference
            depth_map = self.get_depth_map(image_bgr)
            h, w = image_bgr.shape[:2]

            # Clamp indices to image bounds
            cy_a_c = min(max(cy_a, 0), h-1)
            cx_a_c = min(max(cx_a, 0), w-1)
            cy_b_c = min(max(cy_b, 0), h-1)
            cx_b_c = min(max(cx_b, 0), w-1)

            if depth_map is not None:
                depth_a = float(depth_map[cy_a_c, cx_a_c])
                depth_b = float(depth_map[cy_b_c, cx_b_c])
                # If depths are similar → vehicles at same depth plane → mostly lateral distance
                depth_diff = abs(depth_a - depth_b)
                # Scale pixel distance by depth context
                depth_scale = 1.0 + depth_diff * 2.0
                return round(pixel_dist / px_per_m * depth_scale, 2)

        # Pure bounding box heuristic
        return round(pixel_dist / px_per_m, 2)

    def estimate_all_pairs(
        self,
        image_bgr: np.ndarray,
        detections: List[Dict],
    ) -> List[Dict]:
        """
        Estimate distances between all pairs of detected vehicles.

        Returns:
            List of {pair_id, vehicle_a, vehicle_b, distance_m}
        """
        results = []
        depth_map = self.get_depth_map(image_bgr) if self._model else None

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                d_m = self.estimate_distance_between_vehicles(
                    image_bgr,
                    detections[i]["bbox"],
                    detections[j]["bbox"],
                )
                results.append({
                    "pair_id": f"{i}-{j}",
                    "vehicle_a": i,
                    "vehicle_b": j,
                    "distance_m": d_m,
                })

        return results
