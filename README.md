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

Press `q` in the preview window to stop.

### Entity Modeling (JSON)

Build a structured JSON model of all entities, their states, and optionally GPS coordinates.

```bash
uv run python main.py --video input.mp4 --query "person" --entities
```

Output: `input.entities.json`

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
| `--every-n-frames INT` | `2` | Check for scene changes every N frames. On non-keyframes, the previous detection is reused. Higher values skip more frames but may miss fast-moving objects. |
| `--scene-threshold FLOAT` | `0.92` | Scene-change sensitivity (0.0 to 1.0). Controls when SAM3 re-runs inference. A value of `1.0` re-detects on every candidate frame. A value of `0.8` only re-detects on significant scene changes. Lower = more re-detections, higher = more skipping. Best for static cameras (drone, security cam). |
| `--max-reuse INT` | `60` | Safety net: force a re-detection after this many frames even if no scene change is detected. At 30fps, the default of 60 means at least one re-detection every 2 seconds. |

### Output Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | *(auto-generated)* | Output file path. For video mode: defaults to `<input>_tracked.mp4`. For entity mode: defaults to `<input>.entities.json`. |
| `--no-display` | *(off)* | Disable the live OpenCV preview window. Useful for headless/server processing or when you only need the output file. |
| `--camera-id INT` | `0` | Camera device ID for `--camera` mode. Use `1`, `2`, etc. for additional cameras. |

### Entity Modeling Options

These flags are used with `--entities` to produce a JSON entity model instead of (or in addition to) an annotated video.

| Flag | Default | Description |
|------|---------|-------------|
| `--entities` | *(off)* | Enable entity modeling mode. Runs a two-pass pipeline: Pass 1 uses SAM3 to detect and track entities with persistent IDs across frames. Pass 2 loads Gemma 4 to analyze each entity's state (action, pose, spatial context). Outputs a structured JSON file. |
| `--state-every-n INT` | `30` | How often Gemma analyzes entity states, in frames. At 30fps, the default of 30 means states are described once per second. Lower values give more granular state timelines but take longer. |
| `--max-entities INT` | `10` | Maximum entities to analyze per state frame. Limits Gemma inference time when many objects are detected. Entities beyond this cap get `["not analyzed"]` as their state. |
| `--georef PATH` | *(none)* | Path to a georef JSON file containing pixel-to-GPS reference points. When provided, each entity snapshot includes a `gps: [lat, lon]` field computed from the bottom-center of its bounding box via homography transform. Requires 4+ reference points. |

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
                              Scene-change gate                  Path trails
                                                                 Video writer
                                                                 Display
```

- **Reader thread**: decodes video frames into a bounded queue (I/O bound)
- **Inference thread**: runs SAM3 detection on keyframes, assigns persistent entity IDs, skips unchanged scenes (GPU/compute bound)
- **Main thread**: draws annotations (masks, boxes, labels, motion trails), writes output video, shows preview (must be main thread for macOS OpenCV)

### Models

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Scene analysis | `mlx-community/gemma-4-e4b-it-8bit` | ~9 GB | Identifies objects in scene, describes entity states |
| Segmentation | `mlx-community/sam3-bf16` | ~1.7 GB | Object detection, instance segmentation, tracking |

Both models run locally via MLX on Apple Silicon. Memory-managed to fit in 16GB: when both are needed (entity mode), they load/unload sequentially.

## Examples

```bash
# Basic: find and track people, with preview
uv run python main.py --video game.mp4 --query "person"

# Fast: aggressive skipping for static drone footage
uv run python main.py --video drone.mp4 --query "person" --every-n-frames 5 --scene-threshold 0.95

# Headless: no preview, just produce the output file
uv run python main.py --video clip.mp4 --query "car" --no-display

# Auto-detect: let Gemma decide what to segment
uv run python main.py --video scene.mp4

# Entity JSON: structured entity model with states
uv run python main.py --video game.mp4 --query "person" --entities --state-every-n 60

# Full pipeline: entities + GPS + frequent state updates
uv run python main.py --video drone.mp4 --query "person" --entities --georef georef.json --state-every-n 30

# Live camera
uv run python main.py --camera --query "person"
```
