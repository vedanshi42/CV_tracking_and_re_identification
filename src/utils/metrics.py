"""Small processing performance helpers."""

from collections import deque
from time import perf_counter


class FPSMeter:
    """Calculate a smoothed processing FPS from recent frame completion times."""

    def __init__(self, window_size: int = 30):
        self._durations = deque(maxlen=max(1, int(window_size)))
        self._started_at = None

    @property
    def fps(self) -> float:
        """Return recent completed processing FPS, or zero before a frame."""
        elapsed = sum(self._durations)
        return len(self._durations) / elapsed if elapsed > 0 else 0.0

    def start(self) -> None:
        """Start timing one frame's detection and tracking work."""
        self._started_at = perf_counter()

    def stop(self) -> float:
        """Record the current frame duration and return the rolling FPS."""
        if self._started_at is None:
            return 0.0
        duration = perf_counter() - self._started_at
        self._started_at = None
        if duration > 0:
            self._durations.append(duration)
        return self.fps
