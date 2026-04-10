# video-segment: Local Multi-Model Video Understanding Pipeline

## What It Does

video-segment is a local-only video analysis pipeline that chains two AI models together to detect, segment, track, and describe entities in video — then optionally maps them to GPS coordinates. Everything runs on Apple Silicon via MLX. No cloud, no API keys.

## The Pipeline

```
                    ┌──────────────┐
  Video frames  ──> │   Gemma 4    │ ──> object labels ("person", "car", ...)
                    │  (26B 4-bit) │         or entity state descriptions
                    └──────────────┘
                           │
                           v
                    ┌──────────────┐
  Video frames  ──> │    SAM 3     │ ──> bounding boxes + pixel masks + tracking
                    │  (860M bf16) │
                    └──────────────┘
                           │
                           v
                    ┌──────────────┐
  Reference pts ──> │  Homography  │ ──> GPS coordinates per entity
  (optional)        └──────────────┘
```

**Gemma 4** (Google, 26B parameters, 4-bit quantized) handles vision-language tasks: analyzing frames to identify what's in the scene, and describing entity states in natural language.

**SAM 3** (Meta, 860M parameters, bf16) handles detection and segmentation: given a text prompt like "person", it returns bounding boxes and pixel-level masks. We use `predict_multi()` to run the vision backbone once and reuse it across multiple prompts.

## Two Operating Modes

### 1. Annotated Video

Produces a video with colored segmentation masks, bounding boxes, entity IDs, and motion trails overlaid on every frame.

### 2. Entity Model (JSON)

Produces a structured data model of every tracked entity:

```json
{
  "person_001": {
    "timeline": [
      {
        "frame": 0, "timestamp": 0.0,
        "bbox": [120.5, 80.3, 250.1, 400.7],
        "gps": [33.53682, -86.80221],
        "states": ["running", "center of field", "wearing red"]
      }
    ]
  }
}
```

Entity mode runs two passes to stay within 16GB memory: SAM 3 detects and tracks first, then Gemma 4 loads to describe states.

## Performance Architecture

Three concurrent threads keep the GPU saturated:

| Thread | Work | Bottleneck |
|--------|------|------------|
| Reader | Decodes video frames into a bounded queue | Disk I/O |
| Inference | Runs SAM 3 on keyframes, assigns entity IDs | GPU compute |
| Main | Draws annotations, writes output, shows preview | Disk I/O + display |

Additional optimizations:
- **Frame downscaling**: 1080p frames are resized to 640px wide before inference (~3-4x speedup), results are scaled back up for output
- **Smart keyframe detection**: histogram-based scene-change gating skips inference when the frame hasn't meaningfully changed — significant for static cameras
- **Batched multi-label detection**: `predict_multi()` runs the ViT backbone once regardless of how many object types are being detected

## Georeferencing

For stationary camera footage over flat terrain (e.g., drone over a sports field), a homography transform maps pixel coordinates to GPS. The user provides 4+ reference points (pixel position to known GPS coordinate), OpenCV computes a 3x3 transformation matrix, and each entity's foot position (bottom-center of bounding box) is projected to lat/lon on every frame.

## Tech Stack

- **MLX** — Apple's ML framework for Apple Silicon inference
- **mlx-vlm** — MLX ports of vision-language models (Gemma 4, SAM 3)
- **OpenCV** — Video I/O, visualization, homography computation
- **Python 3.12** — Managed with uv
