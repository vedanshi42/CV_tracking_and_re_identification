"""Unit tests for the Day 1 pipeline that do not require a camera or model."""

import unittest
from pathlib import Path

import numpy as np

from main import create_multicamera_source
from src.detection.detector import Detection, PersonDetector
from src.input.multi_camera_source import MultiCameraSource
from src.input.video_source import VideoMetadata, VideoSource
from src.pipeline.day1_pipeline import Day1Pipeline
from src.tracking.tracker import PersonTracker, TrackedObject


class FakeBoxes:
    def __init__(self, xyxy, confidence, class_ids):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape((-1, 4))
        self.conf = np.asarray(confidence, dtype=np.float32)
        self.cls = np.asarray(class_ids, dtype=np.float32)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.options = None

    def predict(self, **options):
        self.options = options
        return [self.results]


class FakeTrackerBackend:
    def __init__(self, track_id=77):
        self.track_id = track_id
        self.last_detections = None

    def update(self, detections, frame):
        self.last_detections = detections
        if len(detections) == 0:
            return np.empty((0, 8), dtype=np.float32)
        row = detections[0]
        return np.asarray(
            [[*row[:4], self.track_id, row[4], row[5], 0]], dtype=np.float32
        )


class FakeVideoSource(VideoSource):
    def __init__(self, frames):
        self.frames = list(frames)
        self.metadata = VideoMetadata(fps=25.0, width=32, height=24, frame_count=len(frames))
        self.opened = False

    def open(self):
        self.opened = True

    def read(self):
        if not self.frames:
            return None
        return self.frames.pop(0)

    def release(self):
        self.opened = False


class FakePipelineTracker:
    def __init__(self, config):
        self.update_count = 0

    def update(self, detections, frame):
        self.update_count += 1
        return [TrackedObject(detections[0].bbox, 1, detections[0].confidence)]


class Day1PipelineTests(unittest.TestCase):
    def test_source_config_builds_named_feeds_and_resolves_file_paths(self):
        config = {
            "sources": [
                {"id": "door", "source_type": "video", "source": "data/door.mp4"},
                {"id": "webcam", "source_type": "webcam", "camera_index": 1},
            ]
        }
        project_root = Path("project_root")
        source = create_multicamera_source(config, project_root)
        self.assertEqual(list(source.sources), ["door", "webcam"])
        self.assertEqual(
            source.sources["door"].source,
            str(project_root / "data/door.mp4"),
        )
        self.assertEqual(source.sources["webcam"].source, 1)

    def test_source_config_rejects_duplicate_ids(self):
        config = {
            "sources": [
                {"id": "same", "source_type": "webcam"},
                {"id": "same", "source_type": "webcam"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            create_multicamera_source(config, Path("."))

    def test_detector_filters_person_and_returns_xyxy_detections(self):
        boxes = FakeBoxes(
            [[1, 2, 10, 20], [4, 5, 9, 15]],
            [0.91, 0.88],
            [0, 2],
        )
        model = FakeModel(FakeResult(boxes))
        detector = PersonDetector(model=model, confidence_threshold=0.2)
        detections = detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (1.0, 2.0, 10.0, 20.0))
        self.assertAlmostEqual(detections[0].confidence, 0.91)
        self.assertEqual(detections[0].class_id, 0)
        self.assertEqual(detections[0].class_name, "person")
        self.assertEqual(model.options["classes"], [0])
        self.assertEqual(model.options["conf"], 0.2)

    def test_detector_handles_empty_detection_result(self):
        model = FakeModel(FakeResult(FakeBoxes([], [], [])))
        detector = PersonDetector(model=model)
        self.assertEqual(detector.detect(np.zeros((24, 32, 3), dtype=np.uint8)), [])

    def test_detector_rejects_empty_frames(self):
        detector = PersonDetector(model=FakeModel(FakeResult(FakeBoxes([], [], []))))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            detector.detect(np.empty((0, 0, 3), dtype=np.uint8))

    def test_tracker_returns_stable_camera_local_ids(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        detection = Detection((1, 2, 10, 20), 0.9, 0, "person")
        first_camera = PersonTracker(backend=FakeTrackerBackend(77))
        second_camera = PersonTracker(backend=FakeTrackerBackend(77))

        first_track = first_camera.update([detection], frame)[0]
        repeated_track = first_camera.update([detection], frame)[0]
        other_camera_track = second_camera.update([detection], frame)[0]

        self.assertEqual(first_track.track_id, 1)
        self.assertEqual(repeated_track.track_id, 1)
        self.assertEqual(other_camera_track.track_id, 1)
        self.assertEqual(first_track.bbox, (1.0, 2.0, 10.0, 20.0))

    def test_pipeline_processes_each_feed_and_releases_sources(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        sources = MultiCameraSource({
            "front": FakeVideoSource([frame]),
            "side": FakeVideoSource([frame]),
        })
        detector = PersonDetector(model=FakeModel(FakeResult(FakeBoxes([], [], []))))
        pipeline = Day1Pipeline(
            source=sources,
            detector=detector,
            tracking_enabled=False,
            display=False,
            save_output=False,
        )

        totals = pipeline.run()

        self.assertEqual(totals, {"front": 1, "side": 1})
        self.assertFalse(sources.sources["front"].opened)
        self.assertFalse(sources.sources["side"].opened)

    def test_pipeline_keeps_one_tracker_per_feed(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        sources = MultiCameraSource({
            "front": FakeVideoSource([frame]),
            "side": FakeVideoSource([frame]),
        })
        boxes = FakeBoxes([[1, 2, 10, 20]], [0.9], [0])
        detector = PersonDetector(model=FakeModel(FakeResult(boxes)))
        created_trackers = []

        def tracker_factory(config):
            tracker = FakePipelineTracker(config)
            created_trackers.append(tracker)
            return tracker

        pipeline = Day1Pipeline(
            source=sources,
            detector=detector,
            tracking_enabled=True,
            display=False,
            save_output=False,
            tracker_factory=tracker_factory,
        )
        totals = pipeline.run()

        self.assertEqual(totals, {"front": 1, "side": 1})
        self.assertEqual(len(created_trackers), 2)
        self.assertIsNot(created_trackers[0], created_trackers[1])
        self.assertEqual([tracker.update_count for tracker in created_trackers], [1, 1])

    def test_pipeline_requires_output_path_when_recording(self):
        with self.assertRaisesRegex(ValueError, "output_path"):
            Day1Pipeline(
                source=MultiCameraSource({"front": FakeVideoSource([])}),
                detector=object(),
                save_output=True,
            )


if __name__ == "__main__":
    unittest.main()
