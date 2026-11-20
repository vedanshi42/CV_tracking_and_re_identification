"""Recorded video file input source."""

import cv2

from ._opencv_source import OpenCVVideoSource


class VideoFileSource(OpenCVVideoSource):
    def __init__(self, path: str, backend: int = cv2.CAP_ANY):
        super().__init__(path, "video file", backend)

    @property
    def eof_is_normal(self) -> bool:
        return True
