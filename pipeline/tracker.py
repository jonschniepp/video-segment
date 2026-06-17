"""Video tracking orchestrator: ties together scene analysis, segmentation, and visualization."""

import json
import os
import tempfile
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .entity_model import Entity, EntitySnapshot, SceneModel
from .entity_tracker import IDTracker
from .georef import bbox_to_gps, compute_homography, load_reference
from .scene_analyzer import SceneAnalyzer
from .segmenter import Detection, Segmenter
from .stabilizer import FrameStabilizer
from .state_analyzer import StateAnalyzer
from .visualizer import PathTracker, draw_detections

# Sentinel value to signal end-of-stream between threads
_SENTINEL = None


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _make_progress(*columns) -> Progress:
    """Create a Rich Progress bar with a standard column set."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        refresh_per_second=10,
    )


# ---------------------------------------------------------------------------
# Scene-change detection (shared by all modes)
# ---------------------------------------------------------------------------

def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalized grayscale histogram for scene-change comparison."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def _scene_changed(hist_a: np.ndarray, hist_b: np.ndarray, threshold: float) -> bool:
    """Return True if two histograms differ enough to indicate a scene change."""
    similarity = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return similarity < threshold


# ---------------------------------------------------------------------------
# Adaptive detection frequency
# ---------------------------------------------------------------------------

def _motion_magnitude(frame: np.ndarray, prev_frame: np.ndarray | None) -> float:
    """Estimate scene motion as mean optical-flow magnitude (0.0 = no motion).

    Uses a downscaled Farneback optical flow for speed.
    """
    if prev_frame is None:
        return 0.0
    scale = 0.25
    h, w = frame.shape[:2]
    small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
    f1 = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (small_w, small_h))
    f2 = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (small_w, small_h))
    flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 8, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
    return float(mag)


def _adaptive_every_n(motion: float, base_every_n: int, min_n: int = 1, max_n: int = 10) -> int:
    """Return a detection interval scaled inversely with motion magnitude.

    High motion → detect more often (lower n).
    Low motion  → detect less often (higher n).
    """
    # motion ~0 → max_n; motion ~5+ → min_n
    scale = max(0.0, 1.0 - motion / 5.0)
    adaptive = int(round(min_n + scale * (max_n - min_n)))
    return min(max(adaptive, min_n), max_n)


# ---------------------------------------------------------------------------
# Async producer/consumer pipeline for process_video
# ---------------------------------------------------------------------------

def _reader_thread(
    cap: cv2.VideoCapture,
    queue: deque,
    max_size: int,
    lock: threading.Lock,
    not_full: threading.Condition,
    not_empty: threading.Condition,
):
    """Read frames from video and push (frame_idx, frame) into the queue."""
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            with not_empty:
                with lock:
                    queue.append(_SENTINEL)
                not_empty.notify()
            break

        with not_full:
            while len(queue) >= max_size:
                not_full.wait()

        with lock:
            queue.append((frame_idx, frame))

        with not_empty:
            not_empty.notify()

        frame_idx += 1


def _inference_thread(
    read_queue: deque,
    write_queue: deque,
    read_lock: threading.Lock,
    read_not_full: threading.Condition,
    read_not_empty: threading.Condition,
    write_lock: threading.Lock,
    write_not_full: threading.Condition,
    write_not_empty: threading.Condition,
    write_max_size: int,
    segmenter: Segmenter,
    labels: list[str],
    every_n: int,
    total: int,
    scene_threshold: float = 0.92,
    max_reuse: int = 60,
    stabilizer: FrameStabilizer | None = None,
    adaptive: bool = True,
    progress_task=None,
    progress=None,
):
    """Pull frames, run SAM3 on keyframes, push results to write queue.

    Keyframe strategy:
      1. First frame — always detect
      2. Adaptive every_n based on optical flow magnitude
      3. Scene-change gate (histogram correlation)
      4. max_reuse safety net
    """
    id_tracker = IDTracker()
    last_detection = None
    last_entity_ids = None
    last_hist = None
    prev_frame: np.ndarray | None = None
    frames_since_detect = 0
    current_every_n = every_n

    while True:
        with read_not_empty:
            while True:
                with read_lock:
                    if len(read_queue) > 0:
                        item = read_queue.popleft()
                        break
                read_not_empty.wait()

        with read_not_full:
            read_not_full.notify()

        if item is _SENTINEL:
            with write_not_empty:
                with write_lock:
                    write_queue.append(_SENTINEL)
                write_not_empty.notify()
            break

        frame_idx, frame = item
        frames_since_detect += 1

        if stabilizer is not None:
            stabilizer.update(frame, frame_idx)

        # Adaptive every_n based on optical flow
        if adaptive and prev_frame is not None:
            motion = _motion_magnitude(frame, prev_frame)
            current_every_n = _adaptive_every_n(motion, every_n)

        run_detect = False
        status_suffix = "skip"

        if last_detection is None:
            run_detect = True
        elif frames_since_detect >= max_reuse:
            run_detect = True
            status_suffix = "forced"
        elif frame_idx % current_every_n == 0:
            curr_hist = _frame_histogram(frame)
            if last_hist is None or _scene_changed(last_hist, curr_hist, scene_threshold):
                run_detect = True
                last_hist = curr_hist
                if stabilizer is not None:
                    stabilizer.reset()

        if run_detect:
            pil_frame = _frame_to_pil(frame)
            last_detection = segmenter.detect(pil_frame, labels)
            last_entity_ids = id_tracker.update(last_detection, frame, frame_idx)
            last_hist = _frame_histogram(frame)
            frames_since_detect = 0
            n_det = len(last_detection.labels)
            status_suffix = f"{n_det} det"

        prev_frame = frame

        if progress is not None and progress_task is not None:
            progress.update(progress_task, advance=1, status=status_suffix)

        with write_not_full:
            while len(write_queue) >= write_max_size:
                write_not_full.wait()

        with write_lock:
            write_queue.append((frame_idx, frame, last_detection, last_entity_ids))

        with write_not_empty:
            write_not_empty.notify()


def process_video(
    video_path: str,
    output_path: str | None = None,
    labels: list[str] | None = None,
    threshold: float = 0.3,
    every_n: int = 2,
    display: bool = True,
    inference_width: int = 640,
    scene_threshold: float = 0.92,
    max_reuse: int = 60,
    stabilize: bool = False,
    adaptive: bool = True,
) -> str:
    """Process a video file with async reader → inference pipeline.

    Two background threads feed the main thread:
      - Reader thread: decodes frames from video (I/O bound)
      - Inference thread: runs SAM3 detection on keyframes (GPU/compute bound)
      - Main thread: draws annotations, writes output, shows preview

    Args:
        video_path: Path to input video.
        output_path: Path for annotated output video. Auto-generated if None.
        labels: List of SAM3 text prompts (e.g. ["person", "backpack"]). None = auto-detect with Gemma 4.
        threshold: SAM3 detection confidence threshold.
        every_n: Base detection interval; adapted per-frame when adaptive=True.
        display: Show live preview window.
        inference_width: Downscale width for SAM3 inference.
        scene_threshold: Histogram correlation threshold (0-1).
        max_reuse: Max frames to reuse a detection before forcing re-detect.
        stabilize: Enable camera-motion stabilization.
        adaptive: Dynamically adjust detection interval based on motion.

    Returns:
        Path to the output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        p = Path(video_path)
        output_path = str(p.with_stem(p.stem + "_tracked"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    analyzer = SceneAnalyzer()
    segmenter = Segmenter(threshold=threshold, inference_width=inference_width)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    if labels:
        print(f"Using labels: {labels}")
    else:
        labels = analyzer.analyze(_frame_to_pil(first_frame))
        print(f"Gemma detected labels: {labels}")
        analyzer.unload()

    segmenter.load()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    READ_QUEUE_SIZE = 16
    WRITE_QUEUE_SIZE = 16

    read_queue: deque = deque()
    write_queue: deque = deque()

    read_lock = threading.Lock()
    read_not_full = threading.Condition()
    read_not_empty = threading.Condition()

    write_lock = threading.Lock()
    write_not_full = threading.Condition()
    write_not_empty = threading.Condition()

    reader = threading.Thread(
        target=_reader_thread,
        args=(cap, read_queue, READ_QUEUE_SIZE, read_lock, read_not_full, read_not_empty),
        daemon=True,
    )

    stabilizer = FrameStabilizer() if stabilize else None

    with _make_progress() as progress:
        task = progress.add_task("Processing video", total=total, status="")

        inferencer = threading.Thread(
            target=_inference_thread,
            args=(
                read_queue, write_queue,
                read_lock, read_not_full, read_not_empty,
                write_lock, write_not_full, write_not_empty,
                WRITE_QUEUE_SIZE,
                segmenter, labels, every_n, total,
                scene_threshold, max_reuse, stabilizer, adaptive,
                task, progress,
            ),
            daemon=True,
        )

        path_tracker = PathTracker(stabilizer=stabilizer)

        try:
            reader.start()
            inferencer.start()

            while True:
                with write_not_empty:
                    while True:
                        with write_lock:
                            if len(write_queue) > 0:
                                item = write_queue.popleft()
                                break
                        write_not_empty.wait()

                with write_not_full:
                    write_not_full.notify()

                if item is _SENTINEL:
                    break

                frame_idx, frame, detection, entity_ids = item

                if detection is not None and entity_ids is not None:
                    path_tracker.update(entity_ids, detection.boxes, frame_idx)
                    annotated = draw_detections(frame, detection, entity_ids, path_tracker, frame_idx)
                elif detection is not None:
                    annotated = draw_detections(frame, detection)
                else:
                    annotated = frame

                vid_writer.write(annotated)

                if display:
                    cv2.imshow("Video Segment", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nStopped by user.")
                        break

            inferencer.join()
            reader.join()
        finally:
            cap.release()
            vid_writer.release()
            if display:
                cv2.destroyAllWindows()
            segmenter.unload()

    print(f"Output saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Camera / stream — async 3-thread pipeline (reader → inference → display)
# ---------------------------------------------------------------------------
# Same architecture as process_video so display never blocks on inference.
# Key differences from video mode:
#   - Queue sizes are tiny (2 frames) to minimise display latency.
#   - Progress bar total=None → indeterminate spinner (stream has no frame count).
#   - Stopping: pressing 'q' releases the capture; the reader thread exits on
#     the next failed cap.read() and propagates the sentinel naturally.
# ---------------------------------------------------------------------------

def process_camera(
    labels: list[str] | None = None,
    threshold: float = 0.3,
    every_n: int = 3,
    camera_source: int | str = 0,
    output_path: str | None = None,
    inference_width: int = 640,
    scene_threshold: float = 0.90,
    max_reuse: int = 30,
    adaptive: bool = True,
    stabilize: bool = False,
) -> str | None:
    """Process a live camera or RTSP stream with a non-blocking 3-thread pipeline.

    Three threads keep display smooth regardless of inference latency:
      - Reader thread:    reads frames from the camera into a tiny bounded queue
      - Inference thread: runs SAM3 on keyframes; pushes annotated results to a
                          second queue; re-uses the last detection on skipped frames
      - Main thread:      pulls results and renders — never waits for inference

    Args:
        labels: SAM3 text prompts. None = auto-detect with Gemma 4 on first frame.
        threshold: SAM3 detection confidence threshold.
        every_n: Base detection interval (adapted when adaptive=True).
        camera_source: Integer device ID for a local camera, or a URL string for
            RTSP/HTTP streams (e.g. "rtsp://192.168.1.10:554/stream").
        output_path: Optional path to save recorded output.
        inference_width: Downscale width for SAM3 inference.
        scene_threshold: Histogram correlation threshold for scene-change gating.
        max_reuse: Max frames to reuse before forcing a re-detect.
        adaptive: Dynamically adjust detection interval based on optical flow.
        stabilize: Enable camera-motion stabilisation for motion trails.

    Returns:
        Output path if recording, else None.
    """
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/stream: {camera_source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # --- Resolve labels before starting threads ---
    if labels:
        resolved_labels = labels
        print(f"Using labels: {resolved_labels}")
    else:
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from camera/stream")
        analyzer = SceneAnalyzer()
        resolved_labels = analyzer.analyze(_frame_to_pil(first_frame))
        print(f"Gemma detected labels: {resolved_labels}")
        analyzer.unload()
        # First frame is consumed by Gemma; stream continues from frame 1 — acceptable.

    segmenter = Segmenter(threshold=threshold, inference_width=inference_width)
    segmenter.load()

    stabilizer = FrameStabilizer() if stabilize else None

    # Small queues — we want the latest frame, not a backlog of stale ones.
    READ_QUEUE_SIZE = 2
    WRITE_QUEUE_SIZE = 2

    read_queue: deque = deque()
    write_queue: deque = deque()

    read_lock = threading.Lock()
    read_not_full = threading.Condition()
    read_not_empty = threading.Condition()

    write_lock = threading.Lock()
    write_not_full = threading.Condition()
    write_not_empty = threading.Condition()

    reader = threading.Thread(
        target=_reader_thread,
        args=(cap, read_queue, READ_QUEUE_SIZE, read_lock, read_not_full, read_not_empty),
        daemon=True,
    )

    path_tracker = PathTracker(stabilizer=stabilizer)

    with _make_progress() as progress:
        # total=None → indeterminate spinner (no frame count for live streams)
        task = progress.add_task("Stream", total=None, status="starting…")

        inferencer = threading.Thread(
            target=_inference_thread,
            args=(
                read_queue, write_queue,
                read_lock, read_not_full, read_not_empty,
                write_lock, write_not_full, write_not_empty,
                WRITE_QUEUE_SIZE,
                segmenter, resolved_labels, every_n, 0,  # total=0 (unknown)
                scene_threshold, max_reuse, stabilizer, adaptive,
                task, progress,
            ),
            daemon=True,
        )

        try:
            reader.start()
            inferencer.start()

            while True:
                # Pull the next annotated frame — blocks only until inference
                # thread has a result ready (typically just the draw time).
                with write_not_empty:
                    while True:
                        with write_lock:
                            if len(write_queue) > 0:
                                item = write_queue.popleft()
                                break
                        write_not_empty.wait()

                with write_not_full:
                    write_not_full.notify()

                if item is _SENTINEL:
                    break

                frame_idx, frame, detection, entity_ids = item

                if detection is not None and entity_ids is not None:
                    path_tracker.update(entity_ids, detection.boxes, frame_idx)
                    annotated = draw_detections(frame, detection, entity_ids, path_tracker, frame_idx)
                elif detection is not None:
                    annotated = draw_detections(frame, detection)
                else:
                    annotated = frame

                cv2.imshow("Video Segment — Camera/Stream (q to quit)", annotated)
                if writer:
                    writer.write(annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    # Release the capture — reader thread gets ret=False on next
                    # cap.read() and sends the sentinel, unwinding cleanly.
                    cap.release()
                    break

            inferencer.join(timeout=3.0)
            reader.join(timeout=3.0)

        finally:
            if cap.isOpened():
                cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            segmenter.unload()

    return output_path


# ---------------------------------------------------------------------------
# Entity modeling (two-pass, with checkpoint/resume + scene-change gating)
# ---------------------------------------------------------------------------

_CHECKPOINT_SUFFIX = ".pass1_checkpoint.json"


def _checkpoint_path(video_path: str) -> str:
    return str(Path(video_path).with_suffix("")) + _CHECKPOINT_SUFFIX


def _save_checkpoint(video_path: str, frame_detections: dict, entity_data: dict):
    """Serialize pass-1 results to a JSON checkpoint file."""
    cp = _checkpoint_path(video_path)
    payload = {
        "frame_detections": {
            str(k): {
                "entity_ids": v["entity_ids"],
                "boxes": [b.tolist() for b in v["boxes"]],
                "scores": v["scores"].tolist(),
                "labels": v["labels"],
            }
            for k, v in frame_detections.items()
        },
        "entity_data": entity_data,
    }
    Path(cp).write_text(json.dumps(payload))


def _load_checkpoint(video_path: str) -> tuple[dict, dict] | None:
    """Load a pass-1 checkpoint if one exists. Returns (frame_detections, entity_data) or None."""
    cp = _checkpoint_path(video_path)
    if not Path(cp).exists():
        return None
    try:
        payload = json.loads(Path(cp).read_text())
        frame_detections = {
            int(k): {
                "entity_ids": v["entity_ids"],
                "boxes": np.array(v["boxes"]),
                "scores": np.array(v["scores"]),
                "labels": v["labels"],
            }
            for k, v in payload["frame_detections"].items()
        }
        return frame_detections, payload["entity_data"]
    except Exception:
        return None


def process_video_entities(
    video_path: str,
    output_json: str | None = None,
    output_video: str | None = None,
    labels: list[str] | None = None,
    threshold: float = 0.3,
    every_n: int = 10,
    state_every_n: int = 30,
    inference_width: int = 640,
    max_entities_per_frame: int = 10,
    georef_path: str | None = None,
    display: bool = True,
    stabilize: bool = False,
    adaptive: bool = True,
    scene_threshold: float = 0.92,
    max_reuse: int = 60,
    resume: bool = True,
    export_csv: bool = True,
    export_coco: bool = True,
) -> tuple[str, str]:
    """Process a video: build a JSON entity model AND produce an annotated tracked video.

    Two-pass approach to manage memory (16GB):
      Pass 1: SAM3 detect + track entities, write annotated video, checkpoint to disk.
      Pass 2: Gemma 4 analyzes entity states on keyframes.

    If a checkpoint from a previous run exists, Pass 1 is skipped and Pass 2
    resumes from where it left off.

    Args:
        video_path: Path to input video.
        output_json: Path for output JSON. Auto-generated if None.
        output_video: Path for annotated output video. Auto-generated if None.
        labels: List of SAM3 text prompts. None = auto-detect with Gemma 4.
        threshold: SAM3 detection confidence threshold.
        every_n: Base SAM3 detection interval (adapted when adaptive=True).
        state_every_n: Gemma state-analysis interval in frames.
        inference_width: Downscale width for SAM3.
        max_entities_per_frame: Max entities to analyze per state frame.
        georef_path: Path to georef JSON with reference points for GPS.
        display: Show live preview window.
        stabilize: Enable camera stabilization.
        adaptive: Dynamically adjust detection interval based on motion.
        scene_threshold: Histogram correlation threshold for scene-change gating.
        max_reuse: Max frames to reuse before forcing re-detect.
        resume: If True, skip pass 1 when a checkpoint exists.
        export_csv: Also write a CSV alongside the JSON.
        export_coco: Also write COCO JSON alongside the JSON.

    Returns:
        Tuple of (json_path, video_path).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    p = Path(video_path)
    if output_json is None:
        output_json = str(p.with_suffix(".entities.json"))
    if output_video is None:
        output_video = str(p.with_stem(p.stem + "_tracked"))

    scene = SceneModel(video_source=video_path, fps=fps, total_frames=total)

    # --- Georeferencing setup ---
    homography = None
    if georef_path:
        ref_data = load_reference(georef_path)
        homography = compute_homography(ref_data["reference_points"])
        print(f"Georeferencing enabled: {len(ref_data['reference_points'])} reference points loaded")

    # -----------------------------------------------------------------------
    # Pass 1: SAM3 detection + ID tracking + annotated video
    # -----------------------------------------------------------------------
    checkpoint = _load_checkpoint(video_path) if resume else None

    if checkpoint is not None:
        frame_detections, raw_entity_data = checkpoint
        print(f"Resuming from checkpoint: {len(frame_detections)} keyframes, skipping Pass 1.")
        # Reconstruct entity objects from saved data
        for eid, edata in raw_entity_data.items():
            entity = Entity(
                entity_id=edata["entity_id"],
                label=edata["label"],
                first_seen_frame=edata["first_seen_frame"],
                last_seen_frame=edata["last_seen_frame"],
            )
            for snap in edata["timeline"]:
                entity.add_snapshot(EntitySnapshot(
                    frame=snap["frame"],
                    timestamp=snap["timestamp"],
                    bbox=snap["bbox"],
                    confidence=snap["confidence"],
                    states=snap["states"],
                    gps=snap.get("gps"),
                    vx=snap.get("vx"),
                    vy=snap.get("vy"),
                ))
            scene.entities[eid] = entity
        scene.frames_analyzed = len(frame_detections)
    else:
        frame_detections = _run_pass1(
            cap=cap,
            fps=fps,
            w=w,
            h=h,
            total=total,
            output_video=output_video,
            video_path=video_path,
            labels=labels,
            threshold=threshold,
            every_n=every_n,
            inference_width=inference_width,
            homography=homography,
            display=display,
            stabilize=stabilize,
            adaptive=adaptive,
            scene_threshold=scene_threshold,
            max_reuse=max_reuse,
            scene=scene,
        )

    # -----------------------------------------------------------------------
    # Pass 2: Gemma state analysis
    # -----------------------------------------------------------------------
    state_frames = sorted([f for f in frame_detections if f % state_every_n == 0])
    if not state_frames and frame_detections:
        state_frames = [min(frame_detections.keys())]

    if state_frames:
        print(f"\n=== Pass 2: State analysis ({len(state_frames)} frames) ===")
        state_analyzer = StateAnalyzer()
        state_analyzer.load()

        with _make_progress() as progress:
            task = progress.add_task("State analysis", total=len(state_frames), status="")

            for si, fidx in enumerate(state_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret:
                    continue

                det = frame_detections[fidx]
                states_list = state_analyzer.analyze_entities_batch(
                    frame, det["boxes"], det["labels"], max_entities=max_entities_per_frame
                )

                for i, eid in enumerate(det["entity_ids"]):
                    entity = scene.entities.get(eid)
                    if entity:
                        for snap in entity.timeline:
                            if snap.frame == fidx:
                                snap.states = states_list[i]
                                break

                progress.update(task, advance=1, status=f"frame {fidx}")

        state_analyzer.unload()

    cap.release()

    # Post-processing: state transitions
    scene.compute_all_transitions()

    # --- Save ---
    scene.save(output_json)

    if export_csv:
        csv_path = str(Path(output_json).with_suffix(".csv"))
        scene.save_csv(csv_path)

    if export_coco:
        coco_path = str(Path(output_json).with_suffix(".coco.json"))
        scene.save_coco(coco_path)

    # Clean up checkpoint now that we're done
    cp = _checkpoint_path(video_path)
    if Path(cp).exists():
        Path(cp).unlink()

    return output_json, output_video


# ---------------------------------------------------------------------------
# Pass 1 implementation (extracted for clarity)
# ---------------------------------------------------------------------------

def _run_pass1(
    cap: cv2.VideoCapture,
    fps: float,
    w: int,
    h: int,
    total: int,
    output_video: str,
    video_path: str,
    labels: list[str] | None,
    threshold: float,
    every_n: int,
    inference_width: int,
    homography,
    display: bool,
    stabilize: bool,
    adaptive: bool,
    scene_threshold: float,
    max_reuse: int,
    scene: SceneModel,
) -> dict:
    """Run pass 1: detect, track, write annotated video, checkpoint to disk.

    Returns frame_detections dict for pass 2.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    stabilizer = FrameStabilizer() if stabilize else None
    id_tracker = IDTracker()
    path_tracker = PathTracker(stabilizer=stabilizer)

    # Resolve labels
    if labels:
        print(f"Using labels: {labels}")
    else:
        analyzer = SceneAnalyzer()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")
        labels = analyzer.analyze(_frame_to_pil(first_frame))
        print(f"Gemma detected labels: {labels}")
        analyzer.unload()

    segmenter = Segmenter(threshold=threshold, inference_width=inference_width)
    segmenter.load()

    frame_detections: dict[int, dict] = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    last_detection = None
    last_entity_ids = None
    last_hist = None
    prev_frame: np.ndarray | None = None
    frames_since_detect = 0
    current_every_n = every_n

    print("\n=== Pass 1: Detection, tracking & video ===")

    try:
        with _make_progress() as progress:
            task = progress.add_task("Pass 1 — detect & track", total=total, status="")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if stabilizer is not None:
                    stabilizer.update(frame, frame_idx)

                frames_since_detect += 1

                # Adaptive interval from optical flow
                if adaptive and prev_frame is not None:
                    motion = _motion_magnitude(frame, prev_frame)
                    current_every_n = _adaptive_every_n(motion, every_n)

                # Scene-change gating
                run_detect = False
                status_suffix = "skip"
                if last_detection is None:
                    run_detect = True
                elif frames_since_detect >= max_reuse:
                    run_detect = True
                    status_suffix = "forced"
                elif frame_idx % current_every_n == 0:
                    curr_hist = _frame_histogram(frame)
                    if last_hist is None or _scene_changed(last_hist, curr_hist, scene_threshold):
                        run_detect = True
                        last_hist = curr_hist
                        if stabilizer is not None:
                            stabilizer.reset()

                if run_detect:
                    pil_frame = _frame_to_pil(frame)
                    last_detection = segmenter.detect(pil_frame, labels)
                    last_entity_ids = id_tracker.update(last_detection, frame, frame_idx)
                    last_hist = _frame_histogram(frame)
                    frames_since_detect = 0
                    n_det = len(last_detection.labels) if last_detection else 0
                    status_suffix = f"{n_det} det"

                    if last_detection and len(last_detection.scores) > 0:
                        timestamp = frame_idx / fps
                        frame_detections[frame_idx] = {
                            "entity_ids": list(last_entity_ids),
                            "boxes": last_detection.boxes.copy(),
                            "scores": last_detection.scores.copy(),
                            "labels": list(last_detection.labels),
                        }

                        for i, eid in enumerate(last_entity_ids):
                            if eid not in scene.entities:
                                scene.entities[eid] = Entity(
                                    entity_id=eid,
                                    label=last_detection.labels[i],
                                )

                            gps = None
                            if homography is not None:
                                lat, lon = bbox_to_gps(homography, last_detection.boxes[i])
                                gps = [lat, lon]

                            # Compute velocity from Kalman filter
                            vx_raw, vy_raw = id_tracker.get_velocity(eid)
                            dt = current_every_n / fps if fps > 0 else 1.0
                            vx = vx_raw / dt if dt > 0 else 0.0
                            vy = vy_raw / dt if dt > 0 else 0.0

                            scene.entities[eid].add_snapshot(EntitySnapshot(
                                frame=frame_idx,
                                timestamp=timestamp,
                                bbox=last_detection.boxes[i].tolist(),
                                confidence=float(last_detection.scores[i]),
                                states=[],
                                gps=gps,
                                vx=vx,
                                vy=vy,
                            ))

                prev_frame = frame

                # Draw + write
                if last_detection is not None and last_entity_ids is not None:
                    path_tracker.update(last_entity_ids, last_detection.boxes, frame_idx)
                    annotated = draw_detections(frame, last_detection, last_entity_ids, path_tracker, frame_idx)
                else:
                    annotated = frame

                vid_writer.write(annotated)

                if display:
                    cv2.imshow("Video Segment", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nStopped by user.")
                        break

                progress.update(task, advance=1, status=status_suffix)
                frame_idx += 1

    finally:
        vid_writer.release()
        if display:
            cv2.destroyAllWindows()

    scene.frames_analyzed = len(frame_detections)
    segmenter.unload()
    print(f"Pass 1 complete: {len(scene.entities)} entities tracked across {scene.frames_analyzed} keyframes")
    print(f"Tracked video saved to: {output_video}")

    # Checkpoint pass-1 results to disk so pass 2 can resume on crash
    entity_data = {}
    for eid, entity in scene.entities.items():
        entity_data[eid] = {
            "entity_id": entity.entity_id,
            "label": entity.label,
            "first_seen_frame": entity.first_seen_frame,
            "last_seen_frame": entity.last_seen_frame,
            "timeline": [
                {
                    "frame": s.frame,
                    "timestamp": s.timestamp,
                    "bbox": s.bbox,
                    "confidence": s.confidence,
                    "states": s.states,
                    **({"gps": s.gps} if s.gps else {}),
                    **({"vx": s.vx, "vy": s.vy} if s.vx is not None else {}),
                }
                for s in entity.timeline
            ],
        }
    _save_checkpoint(video_path, frame_detections, entity_data)
    print(f"Pass 1 checkpoint saved.")

    return frame_detections
