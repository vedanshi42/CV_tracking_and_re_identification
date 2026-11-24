"""Conservative matching between local tracks and global person identities."""

from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Optional

import numpy as np

from .embedding_aggregator import normalize_embedding


@dataclass(frozen=True)
class IdentityObservation:
    camera_id: str
    local_track_id: int
    timestamp: float
    similarity: Optional[float]


@dataclass
class GlobalIdentity:
    global_id: int
    embedding: np.ndarray
    local_tracks: set[tuple[str, int]] = field(default_factory=set)
    camera_ids: set[str] = field(default_factory=set)
    last_seen_timestamp: float = 0.0
    observations: deque[IdentityObservation] = field(default_factory=lambda: deque(maxlen=100))


@dataclass(frozen=True)
class IdentityAssignment:
    global_id: Optional[int]
    status: str
    similarity: Optional[float] = None


class GlobalIdentityManager:
    """Keep a global appearance gallery and link it to local camera tracks.

    Matches use cosine similarity of normalized embeddings. Ambiguous or
    simultaneous conflicting assignments are left unassigned rather than
    guessed. IDs are created only for clear new identities.
    """

    def __init__(
        self,
        match_threshold: float = 0.70,
        ambiguity_margin: float = 0.03,
        allow_simultaneous_cross_camera: bool = False,
        active_track_timeout_seconds: float = 2.0,
        history_size: int = 100,
    ):
        if not -1.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be between -1 and 1")
        if ambiguity_margin < 0.0 or active_track_timeout_seconds < 0.0:
            raise ValueError("ambiguity_margin and timeout must be non-negative")
        self.match_threshold = float(match_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.allow_simultaneous_cross_camera = bool(allow_simultaneous_cross_camera)
        self.active_track_timeout_seconds = float(active_track_timeout_seconds)
        self.history_size = int(history_size)
        self.identities: dict[int, GlobalIdentity] = {}
        self.local_to_global: dict[tuple[str, int], int] = {}
        self._active_tracks: dict[tuple[str, int], float] = {}
        self._next_global_id = 1

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> IdentityAssignment:
        """Associate one local track with a global identity or create one."""
        now = monotonic() if timestamp is None else float(timestamp)
        key = (camera_id, int(local_track_id))
        feature = normalize_embedding(embedding)
        self._prune_active_tracks(now)

        existing_id = self.local_to_global.get(key)
        if existing_id is not None:
            self._update_identity(existing_id, key, feature, now, similarity=None)
            return IdentityAssignment(existing_id, "existing")

        candidates = self._rank_candidates(feature)
        if not candidates:
            return self._create_identity(key, feature, now)

        best_id, best_similarity = candidates[0]
        if best_similarity < self.match_threshold:
            return self._create_identity(key, feature, now)
        if len(candidates) > 1 and best_similarity - candidates[1][1] < self.ambiguity_margin:
            return IdentityAssignment(None, "ambiguous", best_similarity)
        if self._conflicts_with_active_track(best_id, key):
            return IdentityAssignment(None, "active_conflict", best_similarity)

        self._update_identity(best_id, key, feature, now, similarity=best_similarity)
        return IdentityAssignment(best_id, "matched", best_similarity)

    def global_id_for(self, camera_id: str, local_track_id: int) -> Optional[int]:
        return self.local_to_global.get((camera_id, int(local_track_id)))

    def _rank_candidates(self, feature: np.ndarray) -> list[tuple[int, float]]:
        candidates = []
        for global_id, identity in self.identities.items():
            similarity = float(np.dot(feature, identity.embedding))
            candidates.append((global_id, similarity))
        return sorted(candidates, key=lambda candidate: candidate[1], reverse=True)

    def _create_identity(
        self,
        local_key: tuple[str, int],
        feature: np.ndarray,
        timestamp: float,
    ) -> IdentityAssignment:
        global_id = self._next_global_id
        self._next_global_id += 1
        identity = GlobalIdentity(
            global_id=global_id,
            embedding=feature.copy(),
            last_seen_timestamp=timestamp,
            observations=deque(maxlen=self.history_size),
        )
        self.identities[global_id] = identity
        self._update_identity(global_id, local_key, feature, timestamp, similarity=None)
        return IdentityAssignment(global_id, "created")

    def _update_identity(
        self,
        global_id: int,
        local_key: tuple[str, int],
        feature: np.ndarray,
        timestamp: float,
        similarity: Optional[float],
    ) -> None:
        identity = self.identities[global_id]
        identity.embedding = normalize_embedding(identity.embedding + feature)
        identity.local_tracks.add(local_key)
        identity.camera_ids.add(local_key[0])
        identity.last_seen_timestamp = timestamp
        identity.observations.append(
            IdentityObservation(local_key[0], local_key[1], timestamp, similarity)
        )
        self.local_to_global[local_key] = global_id
        self._active_tracks[local_key] = timestamp

    def _conflicts_with_active_track(self, global_id: int, local_key: tuple[str, int]) -> bool:
        if self.allow_simultaneous_cross_camera:
            return False
        for tracked_key in self.identities[global_id].local_tracks:
            if tracked_key == local_key:
                continue
            if tracked_key in self._active_tracks:
                return True
        return False

    def _prune_active_tracks(self, now: float) -> None:
        inactive = [
            local_key
            for local_key, last_seen in self._active_tracks.items()
            if now - last_seen > self.active_track_timeout_seconds
        ]
        for local_key in inactive:
            del self._active_tracks[local_key]
