"""Load project configuration and run the Day 1 detection/tracking pipeline."""

from pathlib import Path
import sys

import cv2
import yaml

from src.detection.detector import PersonDetector
from src.input import MultiCameraSource, StreamSource, VideoFileSource, WebcamSource
from src.pipeline.cv_person_detection_pipeline import CVPersonDetectionPipeline
from src.reid import GlobalIdentityManager, OSNetEmbedder, ReIDService, TrackEmbeddingAggregator


def create_video_source(config: dict):
    """Build one source from its YAML mapping."""
    source_type = config.get("source_type", "video_file")
    backend = config.get("backend")
    backend = cv2.CAP_ANY if backend is None else int(backend)

    if source_type in ("video_file", "video"):
        path = config.get("video_path", config.get("source"))
        if not path:
            raise ValueError("video_path is required for a video file source")
        return VideoFileSource(path, backend)
    if source_type == "webcam":
        return WebcamSource(config.get("camera_index", 0), backend)
    if source_type == "stream":
        return StreamSource(config.get("stream_url", ""), backend)
    raise ValueError(
        f"Unsupported source_type {source_type!r}; "
        "choose 'video_file', 'webcam', or 'stream'"
    )


def create_multicamera_source(input_config: dict, project_root: Path):
    """Create named sources; also accept the earlier single-source YAML form."""
    source_configs = input_config.get("sources")
    if source_configs is None:
        source_configs = [{**input_config, "id": "camera_1"}]
    if not isinstance(source_configs, list) or not source_configs:
        raise ValueError("input.sources must be a non-empty list")

    sources = {}
    for source_config in source_configs:
        if not isinstance(source_config, dict):
            raise ValueError("Each entry in input.sources must be a mapping")
        source_id = source_config.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Each input source must have a non-empty string id")
        if source_id in sources:
            raise ValueError(f"Duplicate input source id: {source_id!r}")

        source_config = {**input_config, **source_config}
        video_path = source_config.get("video_path", source_config.get("source"))
        if video_path and not Path(video_path).is_absolute():
            resolved_path = str(project_root / video_path)
            if "video_path" in source_config:
                source_config["video_path"] = resolved_path
            else:
                source_config["source"] = resolved_path
        sources[source_id] = create_video_source(source_config)
    return MultiCameraSource(sources)


def build_pipeline(config: dict, project_root: Path) -> CVPersonDetectionPipeline:
    """Create pipeline components from the loaded YAML configuration."""
    input_config = config.get("input", {})
    detection_config = config.get("detection", {})
    tracking_config = config.get("tracking", {})
    processing_config = config.get("processing", {})
    reid_config = config.get("reid", {})

    source = create_multicamera_source(input_config, project_root)
    tracker_name = tracking_config.get("name", "bytetrack").lower()
    if tracker_name != "bytetrack":
        raise ValueError(f"Unsupported tracker {tracker_name!r}; this baseline supports 'bytetrack'")
    detector = PersonDetector(
        model_path=detection_config.get("model", "yolo11n.pt"),
        confidence_threshold=detection_config.get("confidence_threshold", 0.35),
        person_class_id=detection_config.get("person_class_id", 0),
        device=detection_config.get("device"),
    )
    output_path = processing_config.get("output_path", "outputs/videos/tracked_output.mp4")
    if output_path and not Path(output_path).is_absolute():
        output_path = str(project_root / output_path)

    reid_service = None
    if reid_config.get("enabled", False):
        checkpoint_path = reid_config.get("model_path")
        if not checkpoint_path:
            raise ValueError("reid.model_path is required when Re-ID is enabled")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        embedder = OSNetEmbedder(
            model_name=reid_config.get("model_name", "osnet_x0_25"),
            model_path=str(checkpoint_path),
            device=reid_config.get("device"),
            num_classes=reid_config.get("num_classes", 751),
            batch_size=reid_config.get("batch_size", 16),
        )
        aggregator = TrackEmbeddingAggregator(
            history_size=reid_config.get("track_embedding_history_size", 10),
            update_interval=reid_config.get("embedding_update_interval", 10),
        )
        identity_manager = GlobalIdentityManager(
            match_threshold=reid_config.get("match_threshold", 0.70),
            ambiguity_margin=reid_config.get("ambiguity_margin", 0.03),
            allow_simultaneous_cross_camera=reid_config.get(
                "allow_simultaneous_cross_camera", False
            ),
            active_track_timeout_seconds=reid_config.get("active_track_timeout_seconds", 2.0),
            history_size=reid_config.get("gallery_history_size", 100),
        )
        reid_service = ReIDService(
            embedder=embedder,
            aggregator=aggregator,
            identity_manager=identity_manager,
            min_crop_width=reid_config.get("min_crop_width", 16),
            min_crop_height=reid_config.get("min_crop_height", 32),
            min_track_embeddings=reid_config.get("min_track_embeddings", 3),
        )

    return CVPersonDetectionPipeline(
        source=source,
        detector=detector,
        tracking_enabled=tracking_config.get("enabled", True),
        tracker_config=tracking_config.get("bytetrack", {}),
        display=processing_config.get("display", True),
        save_output=processing_config.get("save_output", True),
        output_path=output_path,
        output_fps=processing_config.get("fallback_fps", 30.0),
        output_codec=processing_config.get("codec", "mp4v"),
        reid_service=reid_service,
    )


def main() -> int:
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "default.yaml"
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        pipeline = build_pipeline(config, project_root)
        frame_totals = pipeline.run()
        for source_id, count in frame_totals.items():
            print(f"Processed {count} frames from {source_id}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; resources have been released.")
        return 130
    except Exception as exc:
        print(f"Application error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
