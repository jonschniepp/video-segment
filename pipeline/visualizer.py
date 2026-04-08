"""Visualizer: draws segmentation masks, bounding boxes, and labels on video frames."""

import cv2
import numpy as np

from .segmenter import Detection

# Distinct colors for up to 20 object classes (BGR format for OpenCV)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0),
    (255, 0, 128), (0, 255, 128), (128, 128, 255), (255, 128, 128), (128, 255, 128),
    (200, 100, 50), (50, 100, 200), (100, 200, 50), (200, 50, 100), (50, 200, 100),
]

MASK_ALPHA = 0.4


def draw_detections(frame: np.ndarray, detection: Detection) -> np.ndarray:
    """Draw masks, boxes, and labels on a frame.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        detection: Detection result from the segmenter.

    Returns:
        Annotated frame as numpy array (H, W, 3).
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Build a color map: same label gets same color
    unique_labels = list(dict.fromkeys(detection.labels))
    label_color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(unique_labels)}

    for i in range(len(detection.scores)):
        color = label_color[detection.labels[i]]
        score = detection.scores[i]
        label = detection.labels[i]
        x1, y1, x2, y2 = detection.boxes[i].astype(int)

        # Draw mask overlay
        if i < len(detection.masks):
            mask = detection.masks[i]
            # Resize mask to frame dimensions if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (
                np.array(color, dtype=np.float32) * MASK_ALPHA
                + overlay[mask_bool].astype(np.float32) * (1 - MASK_ALPHA)
            ).astype(np.uint8)

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw label with background
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay
