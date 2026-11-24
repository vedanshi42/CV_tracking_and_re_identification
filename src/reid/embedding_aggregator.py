"""Aggregate several appearance observations for each camera-local track."""

from collections import defaultdict, deque
from typing import Hashable, Optional

import numpy as np


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a finite one-dimensional L2-normalized float32 vector."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(vector).all() or norm == 0.0:
        raise ValueError("embedding must be finite and have non-zero length")
    return vector / norm


class TrackEmbeddingAggregator:
    """Keep recent normalized embeddings and average them per local track."""

    def __init__(self, history_size: int = 10, update_interval: int = 10):
        if history_size < 1 or update_interval < 1:
            raise ValueError("history_size and update_interval must be at least one")
        self.history_size = int(history_size)
        self.update_interval = int(update_interval)
        self._history = defaultdict(lambda: deque(maxlen=self.history_size))
        self._last_sample_sequence = {}

    def should_sample(self, camera_id: str, local_track_id: int, frame_sequence: int) -> bool:
        key = (camera_id, int(local_track_id))
        previous = self._last_sample_sequence.get(key)
        return previous is None or frame_sequence - previous >= self.update_interval

    def add(
        self,
        camera_id: str,
        local_track_id: int,
        frame_sequence: int,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """Store a sampled observation and return its normalized track mean."""
        key = (camera_id, int(local_track_id))
        self._history[key].append(normalize_embedding(embedding))
        self._last_sample_sequence[key] = int(frame_sequence)
        return self.get(camera_id, local_track_id)

    def get(self, camera_id: str, local_track_id: int) -> Optional[np.ndarray]:
        history = self._history.get((camera_id, int(local_track_id)))
        if not history:
            return None
        return normalize_embedding(np.mean(np.stack(tuple(history)), axis=0))

    def count(self, camera_id: str, local_track_id: int) -> int:
        return len(self._history.get((camera_id, int(local_track_id)), ()))
