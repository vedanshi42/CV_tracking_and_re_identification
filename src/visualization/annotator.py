"""Draw detection, tracking, and processing information on BGR frames."""

from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from src.detection.detector import Detection
from src.tracking.tracker import TrackedObject


class FrameAnnotator:
    """Create readable bounding boxes and status text for one processed frame."""

    def annotate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
        frame_number: int,
        processing_fps: float,
        source_fps: Optional[float] = None,
        camera_id: Optional[str] = None,
        tracking_enabled: bool = True,
        global_ids: Optional[Mapping[int, int]] = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        objects = tracks if tracking_enabled else detections
        for item in objects:
            x1, y1, x2, y2 = (int(round(value)) for value in item.bbox)
            if tracking_enabled:
                track_id = item.track_id
                color = self._color(track_id)
                global_id = (global_ids or {}).get(track_id)
                identity = f"L:{track_id}" if global_id is None else f"L:{track_id} | G:{global_id}"
                label = f"Person {identity} | {item.confidence:.2f}"
            else:
                color = (40, 210, 40)
                label = f"Person {item.confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_y = max(18, y1 - 8)
            cv2.putText(
                annotated,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        source_fps_text = f"{source_fps:.1f}" if source_fps else "unknown"
        camera_text = f" | Camera: {camera_id}" if camera_id else ""
        status = (
            f"Frame: {frame_number} | People: {len(detections)} | "
            f"Tracks: {len(tracks) if tracking_enabled else 0} | "
            f"Processing: {processing_fps:.1f} FPS | Source: {source_fps_text} FPS"
            f"{camera_text}"
        )
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            status,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated

    @staticmethod
    def _color(track_id: int) -> tuple[int, int, int]:
        """Return a repeatable bright BGR color for a local track id."""
        value = int(track_id) * 37
        return (70 + value % 180, 70 + (value * 3) % 180, 70 + (value * 7) % 180)
