# Day 3: Multi-Camera Person Re-Identification

## Purpose

ByteTrack answers “which detection belongs to the same person in this camera
over time?” Re-ID adds a separate, cautious answer to “could tracks from
different cameras be the same physical person?” The result is a Global Person
ID, distinct from ByteTrack’s camera-local ID.

```text
FramePacket
  -> YOLO person detections
  -> ByteTrack local tracks: camera_1 / L:3
  -> original-frame person crop
  -> OSNet appearance embedding
  -> local-track embedding average
  -> global identity gallery cosine comparison
  -> Global ID (for example G:12), or no assignment if uncertain
```

## Components

`src/reid/crop.py` clips a local track’s `(x1, y1, x2, y2)` pixel box to the
original BGR frame. It returns no crop when the clipped region is invalid or
smaller than the configured size. Cropping the original frame matters because
drawn labels and boxes would contaminate the appearance model input.

`src/reid/osnet_embedder.py` loads OSNet x0.25 once. It uses an official
Market-1501 checkpoint, accepts BGR crops, resizes them to 128x256 pixels
(width x height), converts BGR to RGB, scales values to `[0, 1]`, applies the
ImageNet channel mean and standard deviation, then runs batched inference.
Each output is a 512-dimensional, L2-normalized vector. An embedding encodes
appearance traits such as clothing colour, silhouette, and texture. It is not
a face ID and does not establish identity with certainty.

`src/reid/embedding_aggregator.py` retains a bounded history per
`(camera_id, local_track_id)`. It samples every configured N frames, averages
the normalized features, and normalizes the mean. This makes one poor crop or
blurred frame less influential than relying on one frame alone.

`src/reid/global_identity_manager.py` owns the gallery. A global identity has
a global ID, a representative embedding, associated local tracks, camera IDs,
last-seen time, and bounded observation history. It compares normalized vectors
with their dot product, which equals cosine similarity. A match must clear the
configured threshold and must not be too close to a second candidate. Otherwise
the manager creates a new global ID or leaves an ambiguous case unassigned.

`src/reid/service.py` connects these components after ByteTrack. Detector and
tracker code remain unaware of crops, embeddings, or global IDs.

## Why assignments can be wrong

Similarity can be distorted by illumination, camera colour calibration,
viewpoint, low resolution, occlusion, similar uniforms, and poor boxes. OSNet
was trained on Market-1501, whose cameras and people can differ considerably
from the deployment environment. A global match is a hypothesis, so the system
rejects ambiguous candidates and does not force a concurrent cross-camera match
by default.

## Model choice and alternatives

OSNet is a person Re-ID network designed to combine information at multiple
visual scales. The x0.25 version is intentionally small for a CPU baseline;
larger OSNet variants can improve accuracy but use more compute. StrongSORT or
BoT-SORT can combine appearance features with tracking, but would couple more
logic into the local tracker. Transformer Re-ID models may perform better with
enough hardware, while a deployment-specific fine-tuned model is often the
largest accuracy improvement.

## Configuration

The `reid` block in `configs/default.yaml` controls the feature model, sampling
interval, local history, minimum observations, crop dimensions, gallery match
threshold, ambiguity margin, and same-time cross-camera restriction. Start
with the provided conservative defaults, collect examples from the real cameras,
then tune using observed false matches and missed matches.

## Running and testing

Run the pipeline from the project root:

```powershell
.\.venv\Scripts\python.exe main.py
```

Run deterministic unit tests without a camera or Re-ID checkpoint:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite verifies crop clipping, invalid crop rejection, embedding
normalization, local aggregation, cosine matching, new IDs, ambiguous results,
active camera conflicts, and local-to-global associations.

## Current limits

- Feed reads remain sequential and are not timestamp synchronized.
- The gallery is in memory only and resets when the process restarts.
- Cross-camera transition times and camera topology are not modelled.
- A global identity has no guarantee of being correct; review visual evidence
  before using IDs for decisions about people.
