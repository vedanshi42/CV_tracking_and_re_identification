"""Local webcam input source."""

import cv2

from ._opencv_source import OpenCVVideoSource


class WebcamSource(OpenCVVideoSource):
    def __init__(self, camera_index: int = 0, backend: int = cv2.CAP_ANY):
        super().__init__(int(camera_index), "webcam", backend)
