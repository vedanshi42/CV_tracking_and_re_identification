"""Shared OpenCV capture plumbing for the concrete source classes."""

from typing import Optional

import cv2

from .video_source import VideoMetadata, VideoSource


class OpenCVVideoSource(VideoSource):
    def __init__(self, source, source_name: str, backend: int = cv2.CAP_ANY):
        self.source = source
        self.source_name = source_name
        self.backend = backend
        self.capture: Optional[cv2.VideoCapture] = None
        self.metadata = VideoMetadata()

    def open(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            return
        self.release()
        capture = cv2.VideoCapture(self.source, self.backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open {self.source_name}: {self.source!r}")
        self.capture = capture
        self.metadata = VideoMetadata(
            fps=self._positive_float(cv2.CAP_PROP_FPS),
            width=self._positive_int(cv2.CAP_PROP_FRAME_WIDTH),
            height=self._positive_int(cv2.CAP_PROP_FRAME_HEIGHT),
            frame_count=self._positive_int(cv2.CAP_PROP_FRAME_COUNT),
        )

    def _positive_float(self, prop: int):
        value = float(self.capture.get(prop))
        return value if value > 0 else None

    def _positive_int(self, prop: int):
        value = int(self.capture.get(prop))
        return value if value > 0 else None

    def read(self):
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(f"{self.source_name} is not open")
        ok, frame = self.capture.read()
        if ok:
            return frame
        if self.eof_is_normal:
            return None
        raise RuntimeError(f"Lost connection to {self.source_name}: {self.source!r}")

    @property
    def eof_is_normal(self) -> bool:
        return False

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
