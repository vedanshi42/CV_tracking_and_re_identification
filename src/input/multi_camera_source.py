"""Read frames from several named sources through one small interface."""

from time import monotonic
from typing import Dict, Mapping, Optional

from .video_source import FramePacket, VideoSource


class MultiCameraSource:
    """Own multiple sources and return a tagged frame from each active source.

    A call to ``read`` returns a mapping from source id to its latest packet.
    Finite sources are removed when they reach EOF; ``None`` means all sources
    have ended. Sources are read in configuration order.
    """

    def __init__(self, sources: Mapping[str, VideoSource]):
        if not sources:
            raise ValueError("At least one video source is required")
        if any(not isinstance(source_id, str) or not source_id.strip() for source_id in sources):
            raise ValueError("Every video source must have a non-empty string id")
        self.sources: Dict[str, VideoSource] = dict(sources)
        self._active_ids = []
        self._sequences = {source_id: 0 for source_id in self.sources}
        self._is_open = False

    def open(self) -> None:
        if self._is_open:
            return
        self._active_ids = []
        self._sequences = {source_id: 0 for source_id in self.sources}
        try:
            for source_id, source in self.sources.items():
                source.open()
                self._active_ids.append(source_id)
        except Exception:
            self.release()
            raise
        self._is_open = True

    def read(self) -> Optional[Dict[str, FramePacket]]:
        if not self._is_open:
            raise RuntimeError("MultiCameraSource is not open")

        packets = {}
        still_active = []
        for source_id in self._active_ids:
            source = self.sources[source_id]
            frame = source.read()
            if frame is None:
                source.release()
                continue
            self._sequences[source_id] += 1
            packets[source_id] = FramePacket(
                source_id=source_id,
                frame=frame,
                metadata=source.metadata,
                sequence=self._sequences[source_id],
                timestamp=monotonic(),
            )
            still_active.append(source_id)

        self._active_ids = still_active
        if not packets and not self._active_ids:
            return None
        return packets

    def release(self) -> None:
        for source in self.sources.values():
            source.release()
        self._active_ids = []
        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False
