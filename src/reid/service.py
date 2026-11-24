"""Coordinate crops, embeddings, aggregation, and global identity assignment."""

from typing import Sequence

import numpy as np

from src.tracking.tracker import TrackedObject

from .crop import extract_person_crop
from .embedding_aggregator import TrackEmbeddingAggregator
from .global_identity_manager import GlobalIdentityManager


class ReIDService:
    """Apply Re-ID to tracks without coupling detector or tracker code to it."""

    def __init__(
        self,
        embedder,
        aggregator: TrackEmbeddingAggregator,
        identity_manager: GlobalIdentityManager,
        min_crop_width: int = 16,
        min_crop_height: int = 32,
        min_track_embeddings: int = 3,
    ):
        if min_track_embeddings < 1:
            raise ValueError("min_track_embeddings must be at least one")
        self.embedder = embedder
        self.aggregator = aggregator
        self.identity_manager = identity_manager
        self.min_crop_width = int(min_crop_width)
        self.min_crop_height = int(min_crop_height)
        self.min_track_embeddings = int(min_track_embeddings)

    def identify(
        self,
        camera_id: str,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        frame_sequence: int,
        timestamp: float,
    ) -> dict[int, int]:
        """Return known or newly assigned global IDs keyed by local track ID."""
        sampled_tracks, crops = [], []
        global_ids = {}
        for track in tracks:
            known_id = self.identity_manager.global_id_for(camera_id, track.track_id)
            if known_id is not None:
                global_ids[track.track_id] = known_id
            if not self.aggregator.should_sample(camera_id, track.track_id, frame_sequence):
                continue
            crop = extract_person_crop(
                frame,
                track.bbox,
                min_width=self.min_crop_width,
                min_height=self.min_crop_height,
            )
            if crop is not None:
                sampled_tracks.append(track)
                crops.append(crop)

        for track, embedding in zip(sampled_tracks, self.embedder.embed(crops)):
            representative = self.aggregator.add(
                camera_id,
                track.track_id,
                frame_sequence,
                embedding,
            )
            if self.aggregator.count(camera_id, track.track_id) < self.min_track_embeddings:
                continue
            assignment = self.identity_manager.assign(
                camera_id,
                track.track_id,
                representative,
                timestamp,
            )
            if assignment.global_id is not None:
                global_ids[track.track_id] = assignment.global_id
        return global_ids
