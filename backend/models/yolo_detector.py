"""YOLOv8 Vehicle Detector wrapper."""

import base64
import io
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())

# Colors per class (BGR)
CLASS_COLORS = {
    "car": (0, 200, 255),
    "motorcycle": (255, 100, 0),
    "bus": (0, 255, 100),
    "truck": (200, 0, 255),
}


class YOLODetector:
    """
    Wraps YOLOv8s for vehicle detection in frames and images.
    YOLOv8s (small): mAP=44.9% on COCO — 2× more accurate than YOLOv8n,
    still runs in real-time on CPU.
    """

    def __init__(self, model_name: str = "yolov8s.pt"):
        self.model = YOLO(model_name)
        self.model_name = model_name

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in a single frame.

        Args:
            frame: BGR numpy array (OpenCV format)

        Returns:
            List of dicts: [{id, bbox:[x1,y1,x2,y2], class_name, confidence, centroid}]
        """
        results = self.model(frame, classes=VEHICLE_CLASS_IDS, verbose=False)[0]
        detections = []

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_name = VEHICLE_CLASSES[cls_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "id": i,
                "bbox": [x1, y1, x2, y2],
                "class_name": class_name,
                "confidence": round(conf, 4),
                "centroid": (cx, cy),
                "area": (x2 - x1) * (y2 - y1),
            })

        return detections

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on a frame.

        Args:
            frame: BGR numpy array
            detections: Output from detect_frame()

        Returns:
            Annotated BGR frame
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(cls_name, (0, 255, 255))

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )

            # Draw centroid
            cx, cy = det["centroid"]
            cv2.circle(annotated, (cx, cy), 4, (255, 255, 255), -1)

        return annotated

    def detect_vehicles_in_image(self, image_path: str) -> Dict[str, Any]:
        """
        Full detection on an image file. Returns detections + annotated image as base64.

        Args:
            image_path: Path to image file

        Returns:
            Dict with detections, vehicle_count, annotated_image_b64, inter_vehicle metrics
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        detections = self.detect_frame(frame)
        annotated = self.annotate_frame(frame, detections)

        # Compute inter-vehicle metrics
        iou_max = 0.0
        min_distance_px = float("inf")
        vehicle_pairs = []

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                d_i = detections[i]
                d_j = detections[j]

                iou = _compute_iou(d_i["bbox"], d_j["bbox"])
                dist = _euclidean_distance(d_i["centroid"], d_j["centroid"])

                iou_max = max(iou_max, iou)
                min_distance_px = min(min_distance_px, dist)

                vehicle_pairs.append({
                    "vehicle_a": i,
                    "vehicle_b": j,
                    "iou": round(iou, 4),
                    "centroid_distance_px": round(dist, 2),
                })

        annotated_b64 = _frame_to_base64(annotated)

        return {
            "vehicle_count": len(detections),
            "detections": detections,
            "vehicle_pairs": vehicle_pairs,
            "max_iou": round(iou_max, 4),
            "min_centroid_distance_px": round(min_distance_px, 2) if min_distance_px != float("inf") else None,
            "vehicles_overlapping": iou_max > 0.05,
            "annotated_image_b64": annotated_b64,
        }


# ─── Utility functions ─────────────────────────────────────────────────────────

def _compute_iou(bbox_a: List[int], bbox_b: List[int]) -> float:
    """Compute Intersection-over-Union of two bounding boxes."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def _frame_to_base64(frame: np.ndarray) -> str:
    """Convert BGR OpenCV frame to base64-encoded JPEG string."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")
