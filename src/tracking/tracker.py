"""A small ByteTrack adapter that returns typed, camera-local tracks."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np

from src.detection.detector import Detection


@dataclass(frozen=True)
class TrackedObject:
    """One active track with an absolute-pixel ``(x1, y1, x2, y2)`` box."""

    bbox: Tuple[float, float, float, float]
    track_id: int
    confidence: float
    class_id: int = 0
    class_name: str = "person"


DEFAULT_BYTETRACK_CONFIG = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


class PersonTracker:
    """Maintain ByteTrack state for one camera sequence."""

    def __init__(self, config: Optional[dict] = None, backend: Any = None):
        tracker_config = {**DEFAULT_BYTETRACK_CONFIG, **(config or {})}
        self._boxes_class = None
        if backend is None:
            try:
                from ultralytics.engine.results import Boxes
                from ultralytics.trackers.byte_tracker import BYTETracker

                self._boxes_class = Boxes
                backend = BYTETracker(SimpleNamespace(**tracker_config))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not initialize Ultralytics ByteTrack: {exc}. "
                    "Install the requirements and ensure PyTorch can load."
                ) from exc
        self._backend = backend
        self._local_ids = {}
        self._next_local_id = 1

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[TrackedObject]:
        """Associate current person detections and return active local tracks."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("PersonTracker.update requires a non-empty image frame")

        rows = np.asarray(
            [(*detection.bbox, detection.confidence, detection.class_id) for detection in detections],
            dtype=np.float32,
        ).reshape((-1, 6))
        tracker_input = (
            self._boxes_class(rows, frame.shape[:2]) if self._boxes_class is not None else rows
        )
        try:
            tracked_rows = self._backend.update(tracker_input, frame)
        except Exception as exc:
            raise RuntimeError(f"ByteTrack update failed: {exc}") from exc

        if tracked_rows is None or len(tracked_rows) == 0:
            return []
        tracks = []
        for row in tracked_rows:
            backend_id = int(row[4])
            if backend_id not in self._local_ids:
                self._local_ids[backend_id] = self._next_local_id
                self._next_local_id += 1
            tracks.append(
                TrackedObject(
                    bbox=tuple(float(value) for value in row[:4]),
                    track_id=self._local_ids[backend_id],
                    confidence=float(row[5]),
                    class_id=int(row[6]),
                )
            )
        return tracks
