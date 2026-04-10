"""Entity ID tracker: assigns persistent IDs to detections across frames using IoU matching."""

import numpy as np

from .segmenter import Detection


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class IDTracker:
    """Assigns persistent IDs to detections by matching across frames with IoU.

    Same-label detections are matched greedily by highest IoU. Unmatched
    detections get new IDs. IDs that go unmatched for `max_lost` consecutive
    frames are retired.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

        # Active tracks: entity_id -> {"label": str, "box": np.ndarray, "lost": int}
        self._tracks: dict[str, dict] = {}
        self._counters: dict[str, int] = {}  # label -> next ID number

    def _next_id(self, label: str) -> str:
        count = self._counters.get(label, 0) + 1
        self._counters[label] = count
        return f"{label}_{count:03d}"

    def update(self, detection: Detection) -> list[str]:
        """Match detections to existing tracks and return entity IDs.

        Args:
            detection: Current frame's detections.

        Returns:
            List of entity_id strings, one per detection, in the same order.
        """
        n = len(detection.scores)
        if n == 0:
            # Age out all tracks
            self._age_tracks()
            return []

        assigned_ids = [None] * n
        used_tracks = set()

        # Group detections by label for matching within same class
        label_groups: dict[str, list[int]] = {}
        for i, label in enumerate(detection.labels):
            label_groups.setdefault(label, []).append(i)

        for label, det_indices in label_groups.items():
            # Find active tracks with the same label
            candidate_tracks = [
                (tid, t) for tid, t in self._tracks.items()
                if t["label"] == label and tid not in used_tracks
            ]

            if not candidate_tracks:
                # All new
                for i in det_indices:
                    new_id = self._next_id(label)
                    assigned_ids[i] = new_id
                    self._tracks[new_id] = {
                        "label": label,
                        "box": detection.boxes[i],
                        "lost": 0,
                    }
                    used_tracks.add(new_id)
                continue

            # Compute IoU matrix: detections x tracks
            iou_matrix = np.zeros((len(det_indices), len(candidate_tracks)))
            for di, det_i in enumerate(det_indices):
                for ti, (tid, t) in enumerate(candidate_tracks):
                    iou_matrix[di, ti] = _iou(detection.boxes[det_i], t["box"])

            # Greedy matching: highest IoU first
            matched_dets = set()
            matched_tracks = set()

            while True:
                if iou_matrix.size == 0:
                    break
                best = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[best]
                if best_iou < self.iou_threshold:
                    break

                di, ti = best
                det_i = det_indices[di]
                tid = candidate_tracks[ti][0]

                assigned_ids[det_i] = tid
                self._tracks[tid]["box"] = detection.boxes[det_i]
                self._tracks[tid]["lost"] = 0
                used_tracks.add(tid)

                matched_dets.add(di)
                matched_tracks.add(ti)

                # Zero out matched row and column
                iou_matrix[di, :] = 0
                iou_matrix[:, ti] = 0

            # Create new tracks for unmatched detections
            for di, det_i in enumerate(det_indices):
                if di not in matched_dets:
                    new_id = self._next_id(label)
                    assigned_ids[det_i] = new_id
                    self._tracks[new_id] = {
                        "label": label,
                        "box": detection.boxes[det_i],
                        "lost": 0,
                    }
                    used_tracks.add(new_id)

        self._age_tracks(used_tracks)
        return assigned_ids

    def _age_tracks(self, active_ids: set[str] | None = None):
        """Increment lost counter for unmatched tracks, remove expired ones."""
        active_ids = active_ids or set()
        expired = []
        for tid in self._tracks:
            if tid not in active_ids:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    expired.append(tid)
        for tid in expired:
            del self._tracks[tid]
