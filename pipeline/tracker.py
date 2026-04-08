"""Video tracking orchestrator: ties together scene analysis, segmentation, and visualization."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .scene_analyzer import SceneAnalyzer
from .segmenter import Segmenter
from .visualizer import draw_detections


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def process_video(
    video_path: str,
    output_path: str | None = None,
    query: str | None = None,
    threshold: float = 0.3,
    every_n: int = 2,
    display: bool = True,
) -> str:
    """Process a video file: analyze, segment, and annotate.

    Args:
        video_path: Path to input video.
        output_path: Path for annotated output video. Auto-generated if None.
        query: User query for targeted detection. None = auto-detect with Gemma 4.
        threshold: SAM3 detection confidence threshold.
        every_n: Run detection every N frames (reuse last detection for others).
        display: Show live preview window.

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
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    analyzer = SceneAnalyzer()
    segmenter = Segmenter(threshold=threshold)

    # Phase 1: Use Gemma 4 to determine what to look for (first frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    if query:
        labels = analyzer.analyze(_frame_to_pil(first_frame), query=query)
    else:
        labels = analyzer.analyze(_frame_to_pil(first_frame))

    print(f"Detected labels: {labels}")

    # Unload Gemma to free memory for SAM3
    analyzer.unload()

    # Phase 2: Run SAM3 segmentation across frames
    segmenter.load()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    last_detection = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                pil_frame = _frame_to_pil(frame)
                last_detection = segmenter.detect(pil_frame, labels)
                status = f"Frame {frame_idx}/{total} — {len(last_detection.labels)} detections"
            else:
                status = f"Frame {frame_idx}/{total} — reusing"

            print(f"\r{status}", end="", flush=True)

            if last_detection is not None:
                annotated = draw_detections(frame, last_detection)
            else:
                annotated = frame

            writer.write(annotated)

            if display:
                cv2.imshow("Video Segment", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()
        segmenter.unload()

    print(f"\nOutput saved to: {output_path}")
    return output_path


def process_camera(
    query: str | None = None,
    threshold: float = 0.3,
    every_n: int = 3,
    camera_id: int = 0,
    output_path: str | None = None,
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
    segmenter = Segmenter(threshold=threshold)

    labels = None
    last_detection = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # On first frame (or periodically), analyze scene
            if labels is None:
                pil_frame = _frame_to_pil(frame)
                if query:
                    labels = analyzer.analyze(pil_frame, query=query)
                else:
                    labels = analyzer.analyze(pil_frame)
                print(f"Detected labels: {labels}")
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
