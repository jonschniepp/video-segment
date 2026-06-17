"""Entity ID tracker: assigns persistent IDs to detections across frames.

Improvements over the original greedy IoU tracker:
- Hungarian algorithm (optimal global assignment, not greedy)
- Kalman filter per track (position/velocity prediction between detections)
- Appearance-based re-identification (color histogram) for recovering
  entity identity after occlusion or long gaps
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .segmenter import Detection


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

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


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _box_area(box: np.ndarray) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


# ---------------------------------------------------------------------------
# Appearance embedding (lightweight color histogram)
# ---------------------------------------------------------------------------

def _color_histogram(frame_bgr: np.ndarray | None, box: np.ndarray, bins: int = 32) -> np.ndarray | None:
    """Compute a normalized HSV color histogram for a cropped entity region.

    Returns None if frame is unavailable or crop is empty.
    """
    if frame_bgr is None:
        return None

    import cv2
    x1, y1, x2, y2 = box.astype(int)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins // 2], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s]).astype(np.float32)
    norm = hist.sum()
    return hist / norm if norm > 0 else hist


def _histogram_similarity(h1: np.ndarray | None, h2: np.ndarray | None) -> float:
    """Bhattacharyya coefficient between two histograms (0=no overlap, 1=identical)."""
    if h1 is None or h2 is None or h1.shape != h2.shape:
        return 0.0
    return float(np.sum(np.sqrt(h1 * h2)))


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity model, 2D bounding box)
# ---------------------------------------------------------------------------

class _KalmanTrack:
    """Per-entity Kalman filter tracking [cx, cy, w, h, vx, vy, vw, vh].

    State vector: center-x, center-y, width, height, and their velocities.
    Observation: [cx, cy, w, h].
    """

    def __init__(self, box: np.ndarray):
        import cv2 as _cv2
        self._kf = _cv2.KalmanFilter(8, 4)

        dt = 1.0
        # Transition matrix (constant-velocity model)
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt
        self._kf.transitionMatrix = F

        # Measurement matrix: observe [cx, cy, w, h]
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        self._kf.measurementMatrix = H

        self._kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self._kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self._kf.errorCovPost = np.eye(8, dtype=np.float32)

        # Initialise state from first box
        cx, cy = _center(box)
        w, h = box[2] - box[0], box[3] - box[1]
        self._kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)

    def predict(self) -> np.ndarray:
        """Advance the filter and return the predicted [x1,y1,x2,y2] box."""
        state = self._kf.predict()
        return self._state_to_box(state)

    def update(self, box: np.ndarray) -> np.ndarray:
        """Correct the filter with a new observation."""
        cx, cy = _center(box)
        w, h = box[2] - box[0], box[3] - box[1]
        meas = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        state = self._kf.correct(meas)
        return self._state_to_box(state)

    @staticmethod
    def _state_to_box(state: np.ndarray) -> np.ndarray:
        cx, cy, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        w, h = max(1.0, w), max(1.0, h)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    @property
    def velocity(self) -> tuple[float, float]:
        """Return (vx, vy) in pixels per detection interval."""
        s = self._kf.statePost
        return float(s[4, 0]), float(s[5, 0])


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class IDTracker:
    """Assigns persistent IDs to detections across frames.

    Matching pipeline (per label class):
      1. Build cost matrix as (1 - IoU) between detections and predicted
         Kalman positions, masking entries where center distance exceeds
         max_distance.
      2. Hungarian algorithm finds the globally optimal assignment.
      3. Unmatched detections check the re-ID gallery of recently retired
         tracks for appearance similarity. If a match is found the track
         is resurrected with its original ID.
      4. Remaining unmatched detections get fresh IDs.

    Args:
        iou_threshold: Minimum IoU to accept a Hungarian match.
        max_lost: Frames before a track is retired to the re-ID gallery.
        max_distance: Max center-to-center pixel distance to allow a match.
        reid_frames: How many frames a retired track stays in the re-ID gallery.
        reid_threshold: Minimum histogram similarity to trigger re-ID (0-1).
    """

    def __init__(
        self,
        iou_threshold: float = 0.1,
        max_lost: int = 30,
        max_distance: float = 150.0,
        reid_frames: int = 90,
        reid_threshold: float = 0.55,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.max_distance = max_distance
        self.reid_frames = reid_frames
        self.reid_threshold = reid_threshold

        # Active tracks: entity_id -> {label, box, lost, kalman, hist, last_frame}
        self._tracks: dict[str, dict] = {}
        self._counters: dict[str, int] = {}

        # Re-ID gallery: list of {entity_id, label, hist, retired_frame}
        self._gallery: list[dict] = []

    def _next_id(self, label: str) -> str:
        count = self._counters.get(label, 0) + 1
        self._counters[label] = count
        return f"{label}_{count:03d}"

    def update(
        self,
        detection: Detection,
        frame: np.ndarray | None = None,
        frame_idx: int = 0,
    ) -> list[str]:
        """Match detections to existing tracks and return entity IDs.

        Args:
            detection: Current frame's detections.
            frame: Optional BGR frame for appearance re-ID (can be None).
            frame_idx: Current frame index (used for gallery aging).

        Returns:
            List of entity_id strings, one per detection, in input order.
        """
        n = len(detection.scores)
        if n == 0:
            self._age_tracks(set(), frame_idx)
            return []

        assigned_ids: list[str | None] = [None] * n
        used_tracks: set[str] = set()

        # Group by label for within-class matching
        label_groups: dict[str, list[int]] = {}
        for i, label in enumerate(detection.labels):
            label_groups.setdefault(label, []).append(i)

        for label, det_indices in label_groups.items():
            # Step 1: predict Kalman positions for candidate tracks
            candidate_tracks = [
                (tid, t) for tid, t in self._tracks.items()
                if t["label"] == label and tid not in used_tracks
            ]

            if candidate_tracks:
                # Build cost matrix: (n_dets x n_tracks), masked by distance
                n_d = len(det_indices)
                n_t = len(candidate_tracks)
                cost = np.ones((n_d, n_t), dtype=np.float64)

                for di, det_i in enumerate(det_indices):
                    det_box = detection.boxes[det_i]
                    det_center = _center(det_box)
                    for ti, (tid, t) in enumerate(candidate_tracks):
                        pred_box = t["kalman"].predict()
                        pred_center = _center(pred_box)
                        dist = float(np.linalg.norm(det_center - pred_center))
                        if dist <= self.max_distance:
                            cost[di, ti] = 1.0 - _iou(det_box, pred_box)

                # Hungarian assignment
                row_inds, col_inds = linear_sum_assignment(cost)

                matched_dets: set[int] = set()
                for di, ti in zip(row_inds, col_inds):
                    if cost[di, ti] >= (1.0 - self.iou_threshold):
                        continue  # cost too high — treat as unmatched
                    det_i = det_indices[di]
                    tid, t = candidate_tracks[ti]

                    # Update Kalman with actual observation
                    updated_box = t["kalman"].update(detection.boxes[det_i])

                    # Recompute appearance histogram
                    hist = _color_histogram(frame, detection.boxes[det_i])

                    t["box"] = detection.boxes[det_i]
                    t["hist"] = hist if hist is not None else t.get("hist")
                    t["lost"] = 0
                    t["last_frame"] = frame_idx
                    assigned_ids[det_i] = tid
                    used_tracks.add(tid)
                    matched_dets.add(di)

                unmatched_dets = [
                    det_indices[di]
                    for di in range(len(det_indices))
                    if di not in matched_dets
                ]
            else:
                unmatched_dets = list(det_indices)

            # Step 2: re-ID from gallery for unmatched detections
            still_unmatched = []
            for det_i in unmatched_dets:
                det_hist = _color_histogram(frame, detection.boxes[det_i])
                best_sim, best_entry = 0.0, None

                for entry in self._gallery:
                    if entry["label"] != label:
                        continue
                    sim = _histogram_similarity(det_hist, entry["hist"])
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

                if best_entry is not None and best_sim >= self.reid_threshold:
                    # Resurrect the retired track
                    tid = best_entry["entity_id"]
                    self._gallery.remove(best_entry)
                    kalman = _KalmanTrack(detection.boxes[det_i])
                    hist = _color_histogram(frame, detection.boxes[det_i])
                    self._tracks[tid] = {
                        "label": label,
                        "box": detection.boxes[det_i],
                        "lost": 0,
                        "kalman": kalman,
                        "hist": hist,
                        "last_frame": frame_idx,
                    }
                    assigned_ids[det_i] = tid
                    used_tracks.add(tid)
                else:
                    still_unmatched.append(det_i)

            # Step 3: create fresh tracks for truly new detections
            for det_i in still_unmatched:
                new_id = self._next_id(label)
                kalman = _KalmanTrack(detection.boxes[det_i])
                hist = _color_histogram(frame, detection.boxes[det_i])
                self._tracks[new_id] = {
                    "label": label,
                    "box": detection.boxes[det_i],
                    "lost": 0,
                    "kalman": kalman,
                    "hist": hist,
                    "last_frame": frame_idx,
                }
                assigned_ids[det_i] = new_id
                used_tracks.add(new_id)

        self._age_tracks(used_tracks, frame_idx)
        return assigned_ids

    def get_velocity(self, entity_id: str) -> tuple[float, float]:
        """Return (vx, vy) pixels/detection-interval for a track, or (0, 0)."""
        t = self._tracks.get(entity_id)
        if t is None:
            return 0.0, 0.0
        return t["kalman"].velocity

    def get_predicted_box(self, entity_id: str) -> np.ndarray | None:
        """Return the Kalman-predicted box for a track without advancing the filter."""
        t = self._tracks.get(entity_id)
        if t is None:
            return None
        # peek at state without calling predict() (which advances the filter)
        s = t["kalman"]._kf.statePost
        cx, cy, w, h = s[0, 0], s[1, 0], s[2, 0], s[3, 0]
        w, h = max(1.0, w), max(1.0, h)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def _age_tracks(self, active_ids: set[str], frame_idx: int):
        """Age unmatched tracks; retire expired ones to the re-ID gallery."""
        expired = []
        for tid in self._tracks:
            if tid not in active_ids:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    expired.append(tid)

        for tid in expired:
            t = self._tracks.pop(tid)
            if t.get("hist") is not None:
                self._gallery.append({
                    "entity_id": tid,
                    "label": t["label"],
                    "hist": t["hist"],
                    "retired_frame": frame_idx,
                })

        # Prune stale gallery entries
        self._gallery = [
            e for e in self._gallery
            if frame_idx - e["retired_frame"] <= self.reid_frames
        ]
