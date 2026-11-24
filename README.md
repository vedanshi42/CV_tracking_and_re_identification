# CV Tracking and Re-identification

## Day 1: person detection and camera-local tracking

The Day 1 application reads video from a file, webcam, or network URL; detects people with a pretrained YOLO model; associates detections across frames with ByteTrack; draws boxes, confidence, local track IDs, frame number, and processing FPS; and can display and record annotated output.

Each configured input has a stable ID. The pipeline keeps a separate ByteTrack instance and local ID space for every input, so overlapping numeric IDs from different cameras are expected. This is not cross-camera Re-ID: no global person identity is inferred.

## Setup

Python 3.11 is recommended for the Windows PyTorch and Ultralytics packages.
From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the configured model weights (default `yolo11n.pt`) if they are not already cached. A working PyTorch installation is required by Ultralytics.

## Configure and run

Edit `configs/default.yaml`. `input.sources` is a list, so the same pipeline can
read multiple feeds. Each entry needs a unique `id` and a source type (`video_file`
or the `video` alias, `webcam`, or `stream`).

### Video file

```yaml
input:
  sources:
    - id: lobby
      source_type: video_file
      video_path: data/videos/input.mp4
```

Put the video at that path, then run:

```powershell
python main.py
```

Relative input and output paths are resolved from the project root.

### Webcam

```yaml
input:
  sources:
    - id: laptop
      source_type: webcam
      camera_index: 0
```

Change `camera_index` to select another local or USB camera.

### Network or smartphone camera stream

```yaml
input:
  sources:
    - id: phone
      source_type: stream
      stream_url: "http://192.168.1.5:8080/video"
```

Use a URL exposed by the streaming app and reachable from this computer. The URL and codec must be supported by the OpenCV backend installed on the machine.
If a specific backend is needed, set `backend` to its numeric OpenCV backend ID globally under `input` or separately for a feed.

### Multiple feeds

Add more named entries under `input.sources`, for example a webcam and network stream together. `MultiCameraSource.read()` returns one `FramePacket` per feed per cycle. The packet includes the source ID, BGR frame, source metadata, per-feed sequence number, and monotonic capture timestamp. The current reader reads feeds in configuration order; it does not synchronize them or acquire them concurrently.

### Detection, tracking, and output options

- `detection.model`: Ultralytics model path or name. `yolo11n.pt` is a small pretrained model suitable for a baseline.
- `detection.confidence_threshold`: minimum score passed to ByteTrack. It is set to `0.1` so ByteTrack can use lower-confidence detections in its second association stage.
- `detection.person_class_id`: class index for person, `0` in COCO models.
- `detection.device`: `null` for automatic selection, `cpu`, or a device such as `0` for the first CUDA GPU.
- `tracking.enabled`: turn tracking on or off. `tracking.bytetrack` contains ByteTrack association and lost-track settings.
- `processing.display`: show one annotated window per configured feed.
- `processing.save_output`: enable annotated recording. A single feed uses `output_path`; multiple feeds add each feed ID to the filename.
- `processing.fallback_fps`: output FPS used if a live source does not report a valid FPS. File output uses its source FPS when available.
- `processing.codec`: four-character OpenCV codec, `mp4v` by default.

Press **q** or **Escape** in a display window to stop. Sources, writers, and windows are released during normal shutdown or an error. If an output writer cannot initialize, the application reports the path and continues processing without recording that feed.

## Tests

Run the unit suite without a webcam or downloaded model:

```powershell
python -m unittest discover -s tests -v
```

These tests use mock detector and tracker backends. They validate configuration, detection formatting and empty detections, camera-local ID mapping, and pipeline cleanup. They do not prove that local PyTorch, model weights, a specific camera, stream URL, or video codec works on a particular machine.

## Execution flow

```text
main.py
  ├─ reads configs/default.yaml
  ├─ creates named video sources
  ├─ loads PersonDetector once
  └─ Day1Pipeline.run()
       ├─ MultiCameraSource.read() → {camera_id: FramePacket}
       ├─ PersonDetector.detect(frame) → person detections
       ├─ PersonTracker.update(detections, frame) → local tracks
       ├─ FrameAnnotator.annotate(...) → annotated frame
       ├─ optional display and VideoWriter output
       └─ release sources, writers, and windows
```

### Module responsibilities

- `src/input/`: shared source interface and OpenCV implementations. It only produces frames and metadata; it contains no detection or tracking logic.
- `src/detection/detector.py`: loads YOLO once, filters to the configured person class, and returns absolute-pixel `[x1, y1, x2, y2]` boxes, confidence, class ID, and class name. YOLO internally resizes/letterboxes and normalizes frames for inference, then Ultralytics maps boxes back to original image coordinates.
- `src/tracking/tracker.py`: adapts detections to Ultralytics ByteTrack and returns active boxes with camera-local IDs. ByteTrack matches detections frame-to-frame using motion prediction and box overlap; `track_buffer` controls how many missed frames a track can remain recoverable. Long occlusions or similar overlapping people can still cause IDs to be lost or switch.
- `src/visualization/annotator.py`: draws boxes and labels, using stable colors for local track IDs, plus frame, person, track, source-FPS, and processing-FPSb information.
- `src/utils/metrics.py`: measures time spent in detection and tracking. This processing FPS is not the camera's capture FPS; both are shown separately when source FPS is available.
- `src/pipeline/day1_pipeline.py`: connects the source, detector, per-camera trackers, annotator, optional display, and optional output writers.

## Library choices and alternatives

- **Ultralytics YOLO** provides pretrained weights and a concise Python
  inference API. The small `yolo11n.pt` model favors speed and easy
  experimentation, with lower accuracy than larger models. Faster R-CNN can improve accuracy in some settings but is generally slower; SSD is also lightweight but may miss small or crowded people; RT-DETR is a newer transformer-based alternative with different speed and hardware tradeoffs.
  The detector wrapper isolates model-specific code for later replacement.
- **ByteTrack** is used through Ultralytics' tracker implementation. It uses high- and low-confidence detections to maintain tracks without requiring an appearance embedding model. DeepSORT adds appearance features and may help through occlusions at extra compute and setup cost. BoT-SORT adds motion and optional camera compensation, and OC-SORT changes motion association. The `PersonTracker` wrapper keeps that choice separate from the pipeline.
- **OpenCV VideoCapture** provides one familiar backend interface for files, webcams, and supported network streams. FFmpeg or GStreamer can provide more control over stream protocols and buffering, but add backend-specific setup. The concrete source classes isolate OpenCV so those can be considered later.

## Current limitations

- Live feeds are read sequentially. Slow streams can delay later feeds, and frames are not timestamp-synchronized.
- Track IDs remain local to each camera. Day 3 adds optional appearance-based
  Global IDs; zones, counting, and crowd analysis are future work.
- Output encoding depends on the OpenCV build and installed codecs.
- A network URL can be selected and attempted, but whether it connects depends on reachability, credentials, stream format, and the installed backend.

## Day 3: Multi-camera Re-ID

The pipeline now adds optional cross-camera appearance matching after local
ByteTrack tracking:

```text
Local track -> original-frame crop -> OSNet embedding -> track feature average
            -> cosine similarity gallery -> Global Person ID when confident
```

`L:3` means ByteTrack’s local track 3. `G:12` means the conservative Re-ID
gallery considers this track to be Global Person 12. A local ID from one camera
never becomes the same global identity as the same numeric local ID from another
camera without an appearance match.

The default Re-ID model is OSNet x0.25 with the official Market-1501 checkpoint
at `models/osnet_x0_25_market1501.pth.tar`. The checkpoint is ignored by Git.
The `reid` YAML section controls model device, sampling, feature history,
matching threshold, ambiguity handling, crop minimums, and active cross-camera
conflict protection. See [Day 3 Re-ID notes](docs/day3_reid.md) for the full
architecture, model preprocessing, gallery behavior, limits, and trade-offs.

Torchreid is installed from its official source because it does not publish a
usable PyPI package for this environment. After creating a fresh environment,
install the regular dependencies, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps git+https://github.com/KaiyangZhou/deep-person-reid.git
```

The checkpoint can be downloaded with:

```powershell
.\.venv\Scripts\gdown.exe https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj -O models\osnet_x0_25_market1501.pth.tar
```
