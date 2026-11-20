"""Network video stream input source (HTTP, RTSP, or backend-supported URL)."""

import cv2

from ._opencv_source import OpenCVVideoSource


class StreamSource(OpenCVVideoSource):
    def __init__(self, url: str, backend: int = cv2.CAP_ANY):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("A non-empty stream URL is required")
        super().__init__(url, "network stream", backend)
