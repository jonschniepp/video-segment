"""Visualizer: draws segmentation masks, bounding boxes, labels, and motion paths on video frames."""

from collections import defaultdict

import cv2
import numpy as np

from .segmenter import Detection

# Distinct colors for up to 20 tracked entities (BGR format for OpenCV)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0),
    (255, 0, 128), (0, 255, 128), (128, 128, 255), (255, 128, 128), (128, 255, 128),
    (200, 100, 50), (50, 100, 200), (100, 200, 50), (200, 50, 100), (50, 200, 100),
]

MASK_ALPHA = 0.4
PATH_THICKNESS = 2
PATH_DOT_RADIUS = 3


def _bbox_bottom_center(bbox: np.ndarray) -> tuple[int, int]:
    """Get the bottom-center point of a bounding box (foot position)."""
    x1, y1, x2, y2 = bbox.astype(int)
    return (int((x1 + x2) / 2), int(y2))


class PathTracker:
    """Accumulates bottom-center positions per entity ID for drawing motion trails."""

    def __init__(self, max_length: int = 300):
        self.max_length = max_length
        self.paths: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._id_colors: dict[str, tuple[int, int, int]] = {}
        self._color_idx = 0

    def update(self, entity_ids: list[str], boxes: np.ndarray):
        """Record new positions for tracked entities."""
        for i, eid in enumerate(entity_ids):
            pt = _bbox_bottom_center(boxes[i])
            path = self.paths[eid]
            path.append(pt)
            if len(path) > self.max_length:
                path.pop(0)

            if eid not in self._id_colors:
                self._id_colors[eid] = COLORS[self._color_idx % len(COLORS)]
                self._color_idx += 1

    def get_color(self, entity_id: str) -> tuple[int, int, int]:
        return self._id_colors.get(entity_id, COLORS[0])


def draw_detections(
    frame: np.ndarray,
    detection: Detection,
    entity_ids: list[str] | None = None,
    path_tracker: PathTracker | None = None,
) -> np.ndarray:
    """Draw masks, boxes, labels, and entity motion paths on a frame.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        detection: Detection result from the segmenter.
        entity_ids: Optional list of entity ID strings (one per detection).
            If provided, each entity gets a persistent color and ID label.
        path_tracker: Optional PathTracker for drawing motion trails.

    Returns:
        Annotated frame as numpy array (H, W, 3).
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Determine colors: by entity ID if available, else by label
    if entity_ids and path_tracker:
        id_color = {eid: path_tracker.get_color(eid) for eid in entity_ids}
    else:
        unique_labels = list(dict.fromkeys(detection.labels))
        label_color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(unique_labels)}
        id_color = None

    # Draw motion paths first (underneath everything else)
    if path_tracker:
        for eid, path in path_tracker.paths.items():
            if len(path) < 2:
                continue
            color = path_tracker.get_color(eid)
            # Draw connecting line
            pts = np.array(path, dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=PATH_THICKNESS)
            # Draw dot at most recent position
            cv2.circle(overlay, path[-1], PATH_DOT_RADIUS, color, -1)

    for i in range(len(detection.scores)):
        if entity_ids and id_color:
            color = id_color.get(entity_ids[i], COLORS[0])
            display_id = entity_ids[i]
        else:
            color = label_color[detection.labels[i]]
            display_id = None

        score = detection.scores[i]
        label = detection.labels[i]
        x1, y1, x2, y2 = detection.boxes[i].astype(int)

        # Draw mask overlay
        if i < len(detection.masks):
            mask = detection.masks[i]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (
                np.array(color, dtype=np.float32) * MASK_ALPHA
                + overlay[mask_bool].astype(np.float32) * (1 - MASK_ALPHA)
            ).astype(np.uint8)

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw label with entity ID if available
        if display_id:
            text = f"{display_id} {score:.2f}"
        else:
            text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay
