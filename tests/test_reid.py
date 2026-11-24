"""Re-ID units using deterministic embeddings; no model download is required."""

import unittest

import numpy as np

from src.reid.crop import extract_person_crop
from src.reid.embedding_aggregator import TrackEmbeddingAggregator, normalize_embedding
from src.reid.global_identity_manager import GlobalIdentityManager
from src.reid.service import ReIDService
from src.tracking.tracker import TrackedObject


class FakeEmbedder:
    def __init__(self, embeddings):
        self.embeddings = list(embeddings)
        self.calls = []

    def embed(self, crops):
        self.calls.append(len(crops))
        return [self.embeddings.pop(0) for _ in crops]


class ReIDTests(unittest.TestCase):
    def test_crop_clips_to_image_boundaries(self):
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        crop = extract_person_crop(frame, (-10, 10, 50, 150), min_width=10, min_height=10)
        self.assertEqual(crop.shape, (90, 50, 3))

    def test_crop_rejects_invalid_or_small_boxes(self):
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        self.assertIsNone(extract_person_crop(frame, (30, 40, 20, 60)))
        self.assertIsNone(extract_person_crop(frame, (1, 1, 10, 10), min_width=16, min_height=32))

    def test_embedding_normalization_and_aggregation(self):
        aggregator = TrackEmbeddingAggregator(history_size=3, update_interval=5)
        representative = aggregator.add("cam_a", 1, 1, np.array([3, 4], dtype=np.float32))
        representative = aggregator.add("cam_a", 1, 6, np.array([1, 0], dtype=np.float32))

        self.assertAlmostEqual(float(np.linalg.norm(normalize_embedding(np.array([3, 4])))), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(representative)), 1.0, places=6)
        self.assertEqual(aggregator.count("cam_a", 1), 2)
        self.assertFalse(aggregator.should_sample("cam_a", 1, 9))
        self.assertTrue(aggregator.should_sample("cam_a", 1, 11))

    def test_cosine_matching_creates_then_matches_global_identity(self):
        manager = GlobalIdentityManager(match_threshold=0.8, allow_simultaneous_cross_camera=True)
        first = manager.assign("cam_a", 3, np.array([1.0, 0.0]), timestamp=1.0)
        second = manager.assign("cam_b", 1, np.array([0.99, 0.01]), timestamp=2.0)

        self.assertEqual(first.status, "created")
        self.assertEqual(second.status, "matched")
        self.assertEqual(second.global_id, first.global_id)
        self.assertEqual(manager.global_id_for("cam_a", 3), manager.global_id_for("cam_b", 1))
        self.assertEqual(manager.identities[first.global_id].camera_ids, {"cam_a", "cam_b"})

    def test_low_similarity_creates_a_new_identity(self):
        manager = GlobalIdentityManager(match_threshold=0.8, allow_simultaneous_cross_camera=True)
        first = manager.assign("cam_a", 1, np.array([1.0, 0.0]), timestamp=1.0)
        second = manager.assign("cam_b", 1, np.array([0.0, 1.0]), timestamp=2.0)

        self.assertEqual(second.status, "created")
        self.assertNotEqual(first.global_id, second.global_id)

    def test_ambiguous_match_is_left_unassigned(self):
        manager = GlobalIdentityManager(
            match_threshold=0.7,
            ambiguity_margin=0.05,
            allow_simultaneous_cross_camera=True,
        )
        manager.assign("cam_a", 1, np.array([1.0, 0.0]), timestamp=1.0)
        manager.assign("cam_b", 1, np.array([0.0, 1.0]), timestamp=2.0)
        result = manager.assign("cam_c", 1, np.array([1.0, 1.0]), timestamp=3.0)

        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.global_id)
        self.assertIsNone(manager.global_id_for("cam_c", 1))

    def test_simultaneous_cross_camera_conflict_is_not_forced(self):
        manager = GlobalIdentityManager(match_threshold=0.8, active_track_timeout_seconds=10.0)
        first = manager.assign("cam_a", 1, np.array([1.0, 0.0]), timestamp=1.0)
        conflicting = manager.assign("cam_b", 1, np.array([1.0, 0.0]), timestamp=2.0)

        self.assertEqual(first.status, "created")
        self.assertEqual(conflicting.status, "active_conflict")
        self.assertIsNone(conflicting.global_id)

    def test_service_uses_crops_and_persists_known_global_identity(self):
        frame = np.zeros((80, 60, 3), dtype=np.uint8)
        track = TrackedObject((5, 5, 45, 70), track_id=9, confidence=0.9)
        embedder = FakeEmbedder([np.array([1.0, 0.0]), np.array([1.0, 0.0])])
        service = ReIDService(
            embedder=embedder,
            aggregator=TrackEmbeddingAggregator(update_interval=1),
            identity_manager=GlobalIdentityManager(match_threshold=0.8),
            min_track_embeddings=1,
        )

        ids = service.identify("cam_a", frame, [track], frame_sequence=1, timestamp=1.0)
        known_ids = service.identify("cam_a", frame, [track], frame_sequence=2, timestamp=2.0)

        self.assertEqual(ids, {9: 1})
        self.assertEqual(known_ids, {9: 1})
        self.assertEqual(embedder.calls, [1, 1])


if __name__ == "__main__":
    unittest.main()
