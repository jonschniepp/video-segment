# video-segment

Local video segmentation and entity tracking pipeline running entirely on Apple Silicon. Uses **Gemma 4** for scene understanding and **SAM3** for object detection, segmentation, and tracking — no cloud required.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4), 16GB+ RAM recommended
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
uv sync
```

First run will download ~11GB of models to `~/.cache/huggingface/hub/`.

## Usage

### Annotated Video (default mode)

Process a video file and produce an annotated output with segmentation masks, bounding boxes, entity labels, and motion trails.

```bash
uv run python main.py --video input.mp4 --query "person"
```

Output: `input_tracked.mp4`

### Live Camera

```bash
uv run python main.py --camera --query "person"
```

Press `q` in the preview window to stop. The camera pipeline uses the same scene-change gating and adaptive detection as video mode.

### Entity Modeling (JSON)

Build a structured JSON model of all entities, their states, velocity, and optionally GPS coordinates. Also exports CSV and COCO JSON by default.

```bash
uv run python main.py --video input.mp4 --query "person" --entities
```

Outputs:
- `input.entities.json` — full timeline with states, velocity, and state transitions
- `input.entities.csv` — flat table, one row per entity snapshot (great for spreadsheets)
- `input.entities.coco.json` — COCO detection format for use with standard annotation tools

#### Resuming interrupted runs

Pass 1 (SAM3 detection) is automatically checkpointed to disk when it completes. If the process is interrupted during Pass 2 (Gemma state analysis), re-running the same command will skip Pass 1 and resume from the checkpoint.

```bash
# Force a fresh run, ignoring any checkpoint
uv run python main.py --video input.mp4 --query "person" --entities --no-resume
```

### GPS Georeferencing

Assign real-world GPS coordinates to tracked entities using reference points.

**Step 1 — Pick reference points:**

```bash
uv run python tools/pick_points.py --video input.mp4 --output georef.json
```

Click 4+ known locations in the frame (e.g., corner flags, field markings). Press `q` when done.

**Step 2 — Edit `georef.json`** and fill in the real GPS `[lat, lon]` for each point.

**Step 3 — Run with georeferencing:**

```bash
uv run python main.py --video input.mp4 --query "person" --entities --georef georef.json
```

## CLI Reference

### Input Source (required, pick one)

| Flag | Description |
|------|-------------|
| `--video PATH` | Path to an input video file (mp4, avi, etc.) |
| `--camera` | Use live webcam/camera feed |

### Detection Options

| Flag | Default | Description |
|------|---------|-------------|
| `--query TEXT` | *(auto-detect)* | What to find, e.g. `"person"`, `"yellow vehicles"`. When omitted, Gemma 4 analyzes the first frame and auto-detects all visible object types. When provided, the query is passed directly to SAM3 as a text prompt, skipping Gemma entirely. |
| `--threshold FLOAT` | `0.3` | SAM3 detection confidence threshold. Lower values find more objects but may include false positives. Raise to `0.5`+ for high-precision results. |
| `--inference-width INT` | `640` | Downscale frames to this width before running SAM3. Smaller = faster inference but coarser masks. Set to `0` for full resolution (slower). A 1920px video at width 640 runs ~3-4x faster. |

### Performance Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--every-n-frames INT` | `2` | Base detection interval. When `--no-adaptive` is off (the default), this is a starting point that is tightened automatically during fast motion and relaxed during static scenes. |
| `--no-adaptive` | *(off)* | Disable adaptive detection frequency. Uses the fixed `--every-n-frames` interval instead of adjusting based on optical flow. |
| `--scene-threshold FLOAT` | `0.92` | Scene-change sensitivity (0.0 to 1.0). Controls when SAM3 re-runs inference. A value of `1.0` re-detects on every candidate frame. A value of `0.8` only re-detects on significant scene changes. Lower = more re-detections, higher = more skipping. Best for static cameras (drone, security cam). |
| `--max-reuse INT` | `60` | Safety net: force a re-detection after this many frames even if no scene change is detected. At 30fps, the default of 60 means at least one re-detection every 2 seconds. |

### Output Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | *(auto-generated)* | Output file path. For video mode: defaults to `<input>_tracked.mp4`. For entity mode: defaults to `<input>.entities.json`. |
| `--no-display` | *(off)* | Disable the live OpenCV preview window. Useful for headless/server processing or when you only need the output file. |
| `--camera-id INT` | `0` | Camera device ID for `--camera` mode. Use `1`, `2`, etc. for additional cameras. |

### Entity Modeling Options

These flags are used with `--entities` to produce a JSON entity model.

| Flag | Default | Description |
|------|---------|-------------|
| `--entities` | *(off)* | Enable entity modeling mode. Runs a two-pass pipeline: Pass 1 uses SAM3 to detect and track entities with persistent IDs across frames. Pass 2 loads Gemma 4 to analyze each entity's state (action, pose, spatial context). Outputs JSON, CSV, and COCO JSON. |
| `--state-every-n INT` | `30` | How often Gemma analyzes entity states, in frames. At 30fps, the default of 30 means states are described once per second. Lower values give more granular state timelines but take longer. |
| `--max-entities INT` | `10` | Maximum entities to analyze per state frame. Limits Gemma inference time when many objects are detected. Entities beyond this cap get `["not analyzed"]` as their state. |
| `--georef PATH` | *(none)* | Path to a georef JSON file containing pixel-to-GPS reference points. When provided, each entity snapshot includes a `gps: [lat, lon]` field computed from the bottom-center of its bounding box via homography transform. Requires 4+ reference points. |
| `--stabilize` | *(off)* | Compensate camera drift in motion trails. Recommended for drone footage. |
| `--no-resume` | *(off)* | Ignore any existing Pass 1 checkpoint and re-run from scratch. |
| `--no-csv` | *(off)* | Skip CSV export alongside the JSON. |
| `--no-coco` | *(off)* | Skip COCO JSON export alongside the JSON. |

## Entity JSON Format

```json
{
  "person_001": {
    "entity_id": "person_001",
    "label": "person",
    "first_seen_frame": 0,
    "last_seen_frame": 270,
    "transitions": [
      {
        "frame": 60,
        "timestamp": 2.0,
        "from": ["standing", "near goalpost"],
        "to": ["running", "center of field"]
      }
    ],
    "timeline": [
      {
        "frame": 0,
        "timestamp": 0.0,
        "bbox": [120.5, 80.3, 250.1, 400.7],
        "confidence": 0.91,
        "states": ["standing", "near goalpost", "wearing red"],
        "gps": [33.5368, -86.8022],
        "vx": 0.2,
        "vy": -0.1
      }
    ]
  }
}
```

| Field | Description |
|-------|-------------|
| `transitions` | Automatically detected state changes across the timeline |
| `vx` / `vy` | Horizontal and vertical velocity in pixels/second at each snapshot |
| `gps` | `[lat, lon]` if `--georef` was provided |
| `states` | Comma-separated attributes from Gemma (action, pose, context, appearance) |

## Tools

### `tools/pick_points.py`

Interactive helper for building georef reference files. Opens a video frame, lets you click on known locations, and outputs a JSON template.

```bash
uv run python tools/pick_points.py --video input.mp4 [--frame 0] [--output georef.json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video PATH` | *(required)* | Video file to extract a frame from |
| `--frame INT` | `0` | Which frame to display |
| `--output PATH` | *(stdout)* | Save the reference template to a file instead of printing |

## Architecture

```
Video/Camera
    |
    v
[Reader Thread] --frames--> [Inference Thread] --detections--> [Main Thread]
                              SAM3 + ID Tracker                  Visualizer
                              Kalman prediction                  Path trails
                              Scene-change gate                  Video writer
                              Adaptive interval                  Display
                              Re-ID gallery
```

- **Reader thread**: decodes video frames into a bounded queue (I/O bound)
- **Inference thread**: runs SAM3 on keyframes; uses Kalman-predicted positions for Hungarian matching; applies scene-change gating and adaptive detection intervals (GPU/compute bound)
- **Main thread**: draws annotations (masks, boxes, labels, motion trails), writes output video, shows preview (must be main thread for macOS OpenCV)

### Tracking Pipeline

Each frame in the inference thread goes through:

1. **Adaptive interval** — optical flow magnitude tightens or relaxes the detection interval automatically
2. **Scene-change gate** — histogram correlation against the previous keyframe; skips inference when the scene is static
3. **SAM3 detection** — run only on keyframes that pass the gate (or when the safety-net `max_reuse` triggers)
4. **Hungarian matching** — globally optimal assignment of detections to existing tracks using Kalman-predicted positions
5. **Re-ID gallery** — unmatched detections check recently-retired tracks by color-histogram similarity; entity IDs are recovered when an entity re-enters frame after occlusion

### Models

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Scene analysis | `mlx-community/gemma-4-e4b-it-8bit` | ~9 GB | Identifies objects in scene, describes entity states |
| Segmentation | `mlx-community/sam3-bf16` | ~1.7 GB | Object detection, instance segmentation, tracking |

Both models run locally via MLX on Apple Silicon. Memory-managed to fit in 16GB: when both are needed (entity mode), they load/unload sequentially.

## Examples

```bash
# Basic: find and track people
uv run python main.py --video game.mp4 --query "person"

# Fast: aggressive skipping for static drone footage
uv run python main.py --video drone.mp4 --query "person" --every-n-frames 5 --scene-threshold 0.95

# Headless: no preview, just produce the output file
uv run python main.py --video clip.mp4 --query "car" --no-display

# Auto-detect: let Gemma decide what to segment
uv run python main.py --video scene.mp4

# Entity JSON: structured entity model with states, velocity, transitions
uv run python main.py --video game.mp4 --query "person" --entities --state-every-n 60

# Full pipeline: entities + GPS + frequent state updates
uv run python main.py --video drone.mp4 --query "person" --entities --georef georef.json --state-every-n 30

# Resume after a crash (Pass 1 checkpoint auto-detected)
uv run python main.py --video game.mp4 --query "person" --entities

# Force re-run from scratch, skip CSV export
uv run python main.py --video game.mp4 --query "person" --entities --no-resume --no-csv

# Fixed detection rate (no adaptive interval)
uv run python main.py --video clip.mp4 --query "person" --no-adaptive --every-n-frames 3

# Live camera with stabilized motion trails
uv run python main.py --camera --query "person" --stabilize
```

## Suggested Tests

These scenarios exercise specific features and are good starting points for validating a new run or comparing configurations.

### Tracking quality

```bash
# Verify Hungarian matching and Kalman smoothing on a video with crossing entities.
# Watch for stable IDs that don't swap when two people walk past each other.
uv run python main.py --video game.mp4 --query "person" --threshold 0.25
```

### Re-identification after occlusion

```bash
# In footage where entities exit and re-enter frame (e.g., players going off-field),
# entity IDs should be recovered rather than reassigned.
# Check the entities JSON: the same entity_id should span the gap.
uv run python main.py --video game.mp4 --query "person" --entities --no-display
```

### Adaptive vs. fixed detection rate

```bash
# Compare output quality and processing speed between adaptive and fixed intervals
# on the same video. The adaptive run should use fewer inferences on static scenes.
uv run python main.py --video drone.mp4 --query "person" --no-display
uv run python main.py --video drone.mp4 --query "person" --no-display --no-adaptive --every-n-frames 2
```

### Scene-change sensitivity

```bash
# High threshold (0.98): almost never skips, useful for fast-paced footage
uv run python main.py --video game.mp4 --query "person" --scene-threshold 0.98 --no-display

# Low threshold (0.85): aggressively skips on stable scenes, faster on drone footage
uv run python main.py --video drone.mp4 --query "person" --scene-threshold 0.85 --no-display
```

### Checkpoint resume

```bash
# Run entity mode, then interrupt it (Ctrl-C) during Pass 2.
# Re-run the same command — Pass 1 should be skipped automatically.
uv run python main.py --video game.mp4 --query "person" --entities --no-display
# ^C during Pass 2 ...
uv run python main.py --video game.mp4 --query "person" --entities --no-display
# Should print: "Resuming from checkpoint: N keyframes, skipping Pass 1."
```

### State transitions

```bash
# Run entity mode with frequent state analysis to get enough samples for transitions.
# Inspect the "transitions" array in the output JSON.
uv run python main.py --video game.mp4 --query "person" --entities --state-every-n 15 --no-display
```

### Velocity fields

```bash
# After running entity mode, check vx/vy in the timeline.
# On stationary entities these should be near zero; on running players, non-trivial.
uv run python main.py --video game.mp4 --query "person" --entities --no-display
python3 -c "
import json
data = json.load(open('game.entities.json'))
for eid, e in data['entities'].items():
    snaps = [s for s in e['timeline'] if 'vx' in s]
    if snaps:
        avg_speed = sum((s['vx']**2 + s['vy']**2)**0.5 for s in snaps) / len(snaps)
        print(f'{eid}: avg speed = {avg_speed:.1f} px/s')
"
```

### Export formats

```bash
# Entity mode produces JSON, CSV, and COCO JSON by default.
# Verify all three are present and well-formed.
uv run python main.py --video game.mp4 --query "person" --entities --no-display
ls -lh game.entities.json game.entities.csv game.entities.coco.json
python3 -c "import json; d=json.load(open('game.entities.coco.json')); print(len(d['annotations']), 'annotations,', len(d['categories']), 'categories')"
```

### Batch Gemma state analysis

```bash
# Entity mode with many entities per frame exercises the grid-batching path
# (all crops composited into one image for a single Gemma call).
# Compare wall-clock time against single-entity analysis for the same video.
uv run python main.py --video game.mp4 --query "person" --entities --max-entities 8 --state-every-n 30 --no-display
```

### Full pipeline with GPS

```bash
# Requires a georef.json built from the same video.
uv run python tools/pick_points.py --video drone.mp4 --output georef.json
# Edit georef.json to fill in real GPS coordinates, then:
uv run python main.py --video drone.mp4 --query "person" --entities --georef georef.json \
  --stabilize --state-every-n 30 --no-display
```
