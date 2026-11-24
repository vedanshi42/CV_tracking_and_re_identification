"""Person re-identification components for cross-camera global identities."""

from .crop import extract_person_crop
from .embedding_aggregator import TrackEmbeddingAggregator
from .global_identity_manager import GlobalIdentityManager
from .osnet_embedder import OSNetEmbedder
from .service import ReIDService

__all__ = [
    "OSNetEmbedder",
    "TrackEmbeddingAggregator",
    "GlobalIdentityManager",
    "ReIDService",
    "extract_person_crop",
]
