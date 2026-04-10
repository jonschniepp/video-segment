"""Video tracking orchestrator: ties together scene analysis, segmentation, and visualization."""

import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .entity_model import Entity, EntitySnapshot, SceneModel
from .entity_tracker import IDTracker
from .georef import bbox_to_gps, compute_homography, load_reference
from .scene_analyzer import SceneAnalyzer
from .segmenter import Detection, Segmenter
from .state_analyzer import StateAnalyzer
from .visualizer import PathTracker, draw_detections

# Sentinel value to signal end-of-stream between threads
_SENTINEL = None


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


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
            # Signal end of stream
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


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalized grayscale histogram for scene-change comparison."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def _scene_changed(hist_a: np.ndarray, hist_b: np.ndarray, threshold: float) -> bool:
    """Return True if two histograms differ enough to indicate a scene change."""
    # Correlation: 1.0 = identical, 0.0 = no correlation, -1.0 = inverse
    similarity = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return similarity < threshold


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
):
    """Pull frames, run SAM3 on keyframes, push (frame_idx, frame, detection, entity_ids) to write queue.

    Keyframe strategy: run inference when ANY of these are true:
      1. It's the first frame
      2. Scene change detected (histogram correlation drops below scene_threshold)
      3. max_reuse frames have passed since last detection (safety net)

    When the scene is static, most frames are skipped entirely.
    """
    id_tracker = IDTracker()
    last_detection = None
    last_entity_ids = None
    last_hist = None
    frames_since_detect = 0

    while True:
        # Pull from read queue
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
            # Forward sentinel to write queue
            with write_not_empty:
                with write_lock:
                    write_queue.append(_SENTINEL)
                write_not_empty.notify()
            break

        frame_idx, frame = item
        frames_since_detect += 1

        # Decide whether this frame needs inference
        run_detect = False

        if last_detection is None:
            # First frame — always detect
            run_detect = True
        elif frames_since_detect >= max_reuse:
            # Safety net — don't go too long without re-detecting
            run_detect = True
        elif frame_idx % every_n == 0:
            # Candidate keyframe — check if scene actually changed
            curr_hist = _frame_histogram(frame)
            if last_hist is None or _scene_changed(last_hist, curr_hist, scene_threshold):
                run_detect = True
                last_hist = curr_hist
            # If scene hasn't changed, skip even though it's an every_n frame

        if run_detect:
            pil_frame = _frame_to_pil(frame)
            last_detection = segmenter.detect(pil_frame, labels)
            last_entity_ids = id_tracker.update(last_detection)
            last_hist = _frame_histogram(frame)
            frames_since_detect = 0
            status = f"Frame {frame_idx}/{total} — {len(last_detection.labels)} detections"
        else:
            status = f"Frame {frame_idx}/{total} — reusing (skip)"

        print(f"\r{status}", end="", flush=True)

        # Push to write queue: (frame_idx, frame, detection, entity_ids)
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
    query: str | None = None,
    threshold: float = 0.3,
    every_n: int = 2,
    display: bool = True,
    inference_width: int = 640,
    scene_threshold: float = 0.92,
    max_reuse: int = 60,
) -> str:
    """Process a video file with async reader → inference pipeline.

    Two background threads feed the main thread:
      - Reader thread: decodes frames from video (I/O bound)
      - Inference thread: runs SAM3 detection on keyframes (GPU/compute bound)
      - Main thread: draws annotations, writes output, shows preview (must be
        main thread for macOS OpenCV GUI compatibility)

    Smart keyframe strategy: inference runs only when the scene changes
    (histogram correlation drops below scene_threshold), with a safety net
    of max_reuse frames between forced re-detections.

    Args:
        video_path: Path to input video.
        output_path: Path for annotated output video. Auto-generated if None.
        query: User query for targeted detection. None = auto-detect with Gemma 4.
        threshold: SAM3 detection confidence threshold.
        every_n: Check for scene changes every N frames (default: 2).
        display: Show live preview window.
        inference_width: Downscale width for SAM3 inference.
        scene_threshold: Histogram correlation threshold (0-1). Lower = more
            sensitive to changes. Default 0.92 works well for most video.
        max_reuse: Max frames to reuse a detection before forcing re-detect.

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

    # Phase 1: Determine what to look for
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    if query:
        labels = [query]
        print(f"Using query directly: {labels}")
    else:
        labels = analyzer.analyze(_frame_to_pil(first_frame))
        print(f"Gemma detected labels: {labels}")
        analyzer.unload()

    # Phase 2: Async pipeline — reader → inference → main thread (display + write)
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

    inferencer = threading.Thread(
        target=_inference_thread,
        args=(
            read_queue, write_queue,
            read_lock, read_not_full, read_not_empty,
            write_lock, write_not_full, write_not_empty,
            WRITE_QUEUE_SIZE,
            segmenter, labels, every_n, total,
            scene_threshold, max_reuse,
        ),
        daemon=True,
    )

    path_tracker = PathTracker()

    try:
        reader.start()
        inferencer.start()

        # Main thread: pull from write_queue, annotate with paths, display, write
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
                # Update path history with new positions
                path_tracker.update(entity_ids, detection.boxes)
                annotated = draw_detections(frame, detection, entity_ids, path_tracker)
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

    print(f"\nOutput saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Camera (synchronous — latency matters more than throughput)
# ---------------------------------------------------------------------------

def process_camera(
    query: str | None = None,
    threshold: float = 0.3,
    every_n: int = 3,
    camera_id: int = 0,
    output_path: str | None = None,
    inference_width: int = 640,
) -> str | None:
    """Process live camera feed with segmentation overlay.

    Args:
        query: User query for targeted detection. None = auto-detect with Gemma 4.
        threshold: SAM3 detection confidence threshold.
        every_n: Run detection every N frames.
        camera_id: Camera device ID.
        output_path: Optional path to save recorded output.

    Returns:
        Output path if recording, else None.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    analyzer = SceneAnalyzer()
    segmenter = Segmenter(threshold=threshold, inference_width=inference_width)

    labels = None
    last_detection = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # On first frame, determine what to look for
            if labels is None:
                if query:
                    labels = [query]
                    print(f"Using query directly: {labels}")
                else:
                    pil_frame = _frame_to_pil(frame)
                    labels = analyzer.analyze(pil_frame)
                    print(f"Gemma detected labels: {labels}")
                    analyzer.unload()
                segmenter.load()

            if frame_idx % every_n == 0:
                pil_frame = _frame_to_pil(frame)
                last_detection = segmenter.detect(pil_frame, labels)

            if last_detection is not None:
                annotated = draw_detections(frame, last_detection)
            else:
                annotated = frame

            cv2.imshow("Video Segment — Camera (q to quit)", annotated)
            if writer:
                writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        segmenter.unload()

    return output_path


# ---------------------------------------------------------------------------
# Entity modeling (two-pass, sequential — memory constrained)
# ---------------------------------------------------------------------------

def process_video_entities(
    video_path: str,
    output_json: str | None = None,
    output_video: str | None = None,
    query: str | None = None,
    threshold: float = 0.3,
    every_n: int = 10,
    state_every_n: int = 30,
    inference_width: int = 640,
    max_entities_per_frame: int = 10,
    georef_path: str | None = None,
    display: bool = True,
) -> tuple[str, str]:
    """Process a video: build a JSON entity model AND produce an annotated tracked video.

    Two-pass approach to manage memory (16GB):
      Pass 1: Run SAM3 to detect + track entities, write annotated video with paths.
      Pass 2: Load Gemma 4 and analyze entity states on keyframes.

    Args:
        video_path: Path to input video.
        output_json: Path for output JSON. Auto-generated if None.
        output_video: Path for annotated output video. Auto-generated if None.
        query: User query for targeted detection. None = auto-detect with Gemma 4.
        threshold: SAM3 detection confidence threshold.
        every_n: Run SAM3 detection every N frames.
        state_every_n: Analyze entity states with Gemma every N frames.
        inference_width: Downscale width for SAM3.
        max_entities_per_frame: Max entities to analyze per state frame.
        georef_path: Path to georef JSON with reference points for GPS mapping.
        display: Show live preview window.

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    scene = SceneModel(video_source=video_path, fps=fps, total_frames=total)
    id_tracker = IDTracker()
    path_tracker = PathTracker()

    # --- Determine labels ---
    if query:
        labels = [query]
        print(f"Using query directly: {labels}")
    else:
        analyzer = SceneAnalyzer()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")
        labels = analyzer.analyze(_frame_to_pil(first_frame))
        print(f"Gemma detected labels: {labels}")
        analyzer.unload()

    # --- Georeferencing setup ---
    homography = None
    if georef_path:
        ref_data = load_reference(georef_path)
        homography = compute_homography(ref_data["reference_points"])
        print(f"Georeferencing enabled: {len(ref_data['reference_points'])} reference points loaded")

    # --- Pass 1: SAM3 detection + ID tracking + annotated video ---
    print("\n=== Pass 1: Detection, tracking & video ===")
    segmenter = Segmenter(threshold=threshold, inference_width=inference_width)
    segmenter.load()

    # Store detections for pass 2: frame_idx -> detection info
    frame_detections: dict[int, dict] = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    last_detection = None
    last_entity_ids = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                pil_frame = _frame_to_pil(frame)
                last_detection = segmenter.detect(pil_frame, labels)
                last_entity_ids = id_tracker.update(last_detection)

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

                        scene.entities[eid].add_snapshot(EntitySnapshot(
                            frame=frame_idx,
                            timestamp=timestamp,
                            bbox=last_detection.boxes[i].tolist(),
                            confidence=float(last_detection.scores[i]),
                            states=[],
                            gps=gps,
                        ))

                status = f"Frame {frame_idx}/{total} — {len(last_detection.labels) if last_detection else 0} detections"
            else:
                status = f"Frame {frame_idx}/{total} — reusing"

            print(f"\r{status}", end="", flush=True)

            # Write annotated frame with paths
            if last_detection is not None and last_entity_ids is not None:
                path_tracker.update(last_entity_ids, last_detection.boxes)
                annotated = draw_detections(frame, last_detection, last_entity_ids, path_tracker)
            else:
                annotated = frame

            vid_writer.write(annotated)

            if display:
                cv2.imshow("Video Segment", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break

            frame_idx += 1
    finally:
        vid_writer.release()
        if display:
            cv2.destroyAllWindows()

    scene.frames_analyzed = len(frame_detections)
    segmenter.unload()
    print(f"\nPass 1 complete: {len(scene.entities)} entities tracked across {scene.frames_analyzed} keyframes")
    print(f"Tracked video saved to: {output_video}")

    # --- Pass 2: Gemma state analysis on selected frames ---
    state_frames = sorted([f for f in frame_detections if f % state_every_n == 0])
    if not state_frames and frame_detections:
        state_frames = [min(frame_detections.keys())]

    if state_frames:
        print(f"\n=== Pass 2: State analysis ({len(state_frames)} frames) ===")
        state_analyzer = StateAnalyzer()
        state_analyzer.load()

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

            print(f"\rPass 2: {si + 1}/{len(state_frames)} frames analyzed", end="", flush=True)

        state_analyzer.unload()
        print()

    cap.release()

    # --- Save ---
    scene.save(output_json)
    return output_json, output_video
