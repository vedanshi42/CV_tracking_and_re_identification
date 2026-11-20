"""Connect configured frame sources, YOLO detection, ByteTrack, and output."""

import re
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2

from src.input.multi_camera_source import MultiCameraSource
from src.tracking.tracker import PersonTracker
from src.utils.metrics import FPSMeter
from src.visualization.annotator import FrameAnnotator


class Day1Pipeline:
    """Run detection and independent single-camera tracking on every feed."""

    def __init__(
        self,
        source: MultiCameraSource,
        detector,
        tracking_enabled: bool = True,
        tracker_config: Optional[dict] = None,
        display: bool = True,
        save_output: bool = False,
        output_path: Optional[str] = None,
        output_fps: float = 30.0,
        output_codec: str = "mp4v",
        annotator: Optional[FrameAnnotator] = None,
        tracker_factory: Optional[Callable] = None,
    ):
        if save_output and not output_path:
            raise ValueError("output_path is required when save_output is enabled")
        if output_fps <= 0:
            raise ValueError("output_fps must be greater than zero")
        if len(output_codec) != 4:
            raise ValueError("output_codec must be a four-character OpenCV codec")

        self.source = source
        self.detector = detector
        self.tracking_enabled = tracking_enabled
        self.tracker_config = tracker_config or {}
        self.display = display
        self.save_output = save_output
        self.output_path = Path(output_path) if output_path else None
        self.output_fps = float(output_fps)
        self.output_codec = output_codec
        self.annotator = annotator or FrameAnnotator()
        self.tracker_factory = tracker_factory or PersonTracker
        self._trackers = {}
        self._fps_meters = {}
        self._writers = {}
        self._writer_failures = set()

    def run(self) -> dict:
        """Process feeds until they end or the user presses q/Escape."""
        frame_totals = {source_id: 0 for source_id in self.source.sources}
        stop_requested = False
        try:
            with self.source:
                for source_id, video_source in self.source.sources.items():
                    print(f"Opened {source_id}: {video_source.metadata}")

                while not stop_requested:
                    packets = self.source.read()
                    if packets is None:
                        break
                    for source_id, packet in packets.items():
                        fps_meter = self._fps_meters.setdefault(source_id, FPSMeter())
                        fps_meter.start()

                        detections = self.detector.detect(packet.frame)
                        tracks = []
                        if self.tracking_enabled:
                            tracker = self._trackers.get(source_id)
                            if tracker is None:
                                tracker = self.tracker_factory(self.tracker_config)
                                self._trackers[source_id] = tracker
                            tracks = tracker.update(detections, packet.frame)

                        processing_fps = fps_meter.stop()

                        annotated = self.annotator.annotate(
                            frame=packet.frame,
                            detections=detections,
                            tracks=tracks,
                            frame_number=packet.sequence,
                            processing_fps=processing_fps,
                            source_fps=packet.metadata.fps,
                            camera_id=source_id,
                            tracking_enabled=self.tracking_enabled,
                        )
                        frame_totals[source_id] += 1
                        self._write_frame(source_id, annotated, packet.metadata.fps)

                        if self.display:
                            cv2.imshow(f"CV Tracking - {source_id}", annotated)
                    if self.display:
                        key = cv2.waitKey(1) & 0xFF
                        stop_requested = key in (ord("q"), 27)
        finally:
            self._release_writers()
            if self.display:
                cv2.destroyAllWindows()
        return frame_totals

    def _write_frame(self, source_id: str, frame, source_fps: Optional[float]) -> None:
        if not self.save_output or source_id in self._writer_failures:
            return
        writer = self._writers.get(source_id)
        if writer is None:
            output_path = self._output_path_for(source_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fps = source_fps if source_fps and source_fps > 0 else self.output_fps
            try:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*self.output_codec),
                    float(fps),
                    (width, height),
                )
            except cv2.error as exc:
                self._writer_failures.add(source_id)
                print(
                    f"Could not initialize output video writer for {source_id!r} "
                    f"at {output_path}: {exc}; continuing without recording this feed.",
                    file=sys.stderr,
                )
                return
            if not writer.isOpened():
                writer.release()
                self._writer_failures.add(source_id)
                print(
                    f"Could not initialize output video writer for {source_id!r} "
                    f"at {output_path}; continuing without recording this feed.",
                    file=sys.stderr,
                )
                return
            self._writers[source_id] = writer
        writer.write(frame)

    def _output_path_for(self, source_id: str) -> Path:
        if len(self.source.sources) == 1:
            return self.output_path
        safe_source_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)
        return self.output_path.with_name(
            f"{self.output_path.stem}_{safe_source_id}{self.output_path.suffix}"
        )

    def _release_writers(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
