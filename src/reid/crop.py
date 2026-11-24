"""Extract valid person crops from original OpenCV BGR frames."""

from typing import Optional, Sequence

import numpy as np


def extract_person_crop(
    frame: np.ndarray,
    bbox: Sequence[float],
    min_width: int = 16,
    min_height: int = 32,
) -> Optional[np.ndarray]:
    """Return a BGR crop for an ``(x1, y1, x2, y2)`` pixel bounding box.

    Coordinates use an inclusive top-left corner and exclusive bottom-right
    boundary after clipping, matching NumPy slicing. The returned crop is a
    view of ``frame`` when possible; callers must not modify it in place.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
        raise ValueError("frame must be a non-empty image array")
    if frame.size == 0 or len(bbox) != 4:
        return None
    if min_width < 1 or min_height < 1:
        raise ValueError("minimum crop dimensions must be positive")

    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < min_width or y2 - y1 < min_height:
        return None
    return frame[y1:y2, x1:x2]
