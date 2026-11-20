"""YOLO-based person detection, independent of video input."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One person detection with an absolute-pixel ``(x1, y1, x2, y2)`` box."""

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class PersonDetector:
    """Load one Ultralytics YOLO model and return person detections per frame."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        person_class_id: int = 0,
        device: Optional[str] = None,
        model: Any = None,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = float(confidence_threshold)
        self.person_class_id = int(person_class_id)
        self.device = device

        if model is None:
            try:
                from ultralytics import YOLO

                model = YOLO(model_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load YOLO model {model_path!r}: {exc}. "
                    "Install the requirements and ensure PyTorch can load."
                ) from exc
        self.model = model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return person detections for one non-empty OpenCV BGR frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("PersonDetector.detect requires a non-empty image frame")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR frame")

        predict_options = {
            "source": frame,
            "conf": self.confidence_threshold,
            "classes": [self.person_class_id],
            "verbose": False,
        }
        if self.device is not None:
            predict_options["device"] = self.device
        try:
            results = self.model.predict(**predict_options)
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confidences = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        class_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

        detections = []
        for bbox, confidence, class_id in zip(xyxy, confidences, class_ids):
            class_id = int(class_id)
            if class_id != self.person_class_id:
                continue
            detections.append(
                Detection(
                    bbox=tuple(float(value) for value in bbox[:4]),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name="person",
                )
            )
        return detections
