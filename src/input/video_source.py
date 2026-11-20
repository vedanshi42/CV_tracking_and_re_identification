"""Common interface and metadata for frame-producing video sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Properties reported by a source; unknown values are ``None``."""

    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None


@dataclass(frozen=True)
class FramePacket:
    """A frame tagged with the source identity needed by downstream logic."""

    source_id: str
    frame: np.ndarray
    metadata: VideoMetadata
    sequence: int
    timestamp: float


class VideoSource(ABC):
    """A minimal source of BGR image frames.

    ``read`` returns ``None`` only when a finite source reaches EOF. Sources
    that fail while opening or unexpectedly stop delivering frames raise an
    exception instead.
    """

    metadata: VideoMetadata

    @abstractmethod
    def open(self) -> None:
        """Connect to the source or raise an informative exception."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or ``None`` at end-of-video."""

    @abstractmethod
    def release(self) -> None:
        """Release source resources; safe to call more than once."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False
