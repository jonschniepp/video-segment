"""Frame stabilizer: compensates camera drift using ORB feature matching + homography."""

import cv2
import numpy as np


class FrameStabilizer:
    """Computes cumulative homography transforms to compensate camera drift.

    For each frame, detects ORB features, matches against the previous frame,
    and accumulates a homography that maps current-frame pixel coordinates
    into frame-0 (reference) coordinate space.
    """

    def __init__(self, max_features: int = 500, match_ratio: float = 0.75):
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._match_ratio = match_ratio

        self._prev_gray: np.ndarray | None = None
        self._prev_kp = None
        self._prev_des = None

        self._cumulative_H = np.eye(3, dtype=np.float64)
        self._homographies: dict[int, np.ndarray] = {}
        self._inverse_cache: dict[int, np.ndarray] = {}
        self._frame_count = 0

    def update(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process a new frame and store its cumulative homography.

        Args:
            frame: BGR image (H, W, 3).
            frame_idx: Frame index for later lookups.

        Returns:
            3x3 cumulative homography mapping current-frame coords to reference-frame coords.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)

        if self._prev_des is None or des is None or len(kp) < 4:
            # First frame or insufficient features — identity transform
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            self._homographies[frame_idx] = self._cumulative_H.copy()
            self._frame_count += 1
            return self._cumulative_H.copy()

        # Match current frame descriptors against previous frame
        matches = self._bf.knnMatch(des, self._prev_des, k=2)

        # Lowe's ratio test
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self._match_ratio * n.distance:
                    good.append(m)

        if len(good) < 4:
            # Not enough matches — assume no camera motion this frame
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_des = des
            self._homographies[frame_idx] = self._cumulative_H.copy()
            self._frame_count += 1
            return self._cumulative_H.copy()

        # Extract matched point coordinates
        curr_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        prev_pts = np.float32([self._prev_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Homography: maps current-frame points to previous-frame space
        H, mask = cv2.findHomography(curr_pts, prev_pts, cv2.RANSAC, 5.0)

        if H is not None:
            self._cumulative_H = self._cumulative_H @ H

            # Re-normalize every 100 frames to prevent float drift
            if self._frame_count % 100 == 0 and self._cumulative_H[2, 2] != 0:
                self._cumulative_H /= self._cumulative_H[2, 2]

        self._homographies[frame_idx] = self._cumulative_H.copy()
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_des = des
        self._frame_count += 1

        return self._cumulative_H.copy()

    def warp_points_to_ref(self, points: list[tuple[int, int]], frame_idx: int) -> list[tuple[int, int]]:
        """Warp pixel coordinates from a given frame into reference-frame space.

        Args:
            points: List of (x, y) pixel coordinates in the given frame.
            frame_idx: The frame these points belong to.

        Returns:
            List of (x, y) coordinates in reference-frame space.
        """
        H = self._homographies.get(frame_idx)
        if H is None or len(points) == 0:
            return list(points)

        pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        return [(int(round(p[0])), int(round(p[1]))) for p in warped.reshape(-1, 2)]

    def warp_points_from_ref(self, points: list[tuple[int, int]], frame_idx: int) -> list[tuple[int, int]]:
        """Warp reference-frame coordinates back to a given frame's pixel space.

        Args:
            points: List of (x, y) coordinates in reference-frame space.
            frame_idx: The target frame to warp into.

        Returns:
            List of (x, y) pixel coordinates in the target frame.
        """
        if frame_idx not in self._inverse_cache:
            H = self._homographies.get(frame_idx)
            if H is None:
                return list(points)
            H_inv = np.linalg.inv(H)
            self._inverse_cache[frame_idx] = H_inv

        H_inv = self._inverse_cache[frame_idx]
        if len(points) == 0:
            return []

        pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H_inv)
        return [(int(round(p[0])), int(round(p[1]))) for p in warped.reshape(-1, 2)]

    def reset(self):
        """Reset accumulation — call on scene changes."""
        self._cumulative_H = np.eye(3, dtype=np.float64)
        self._prev_gray = None
        self._prev_kp = None
        self._prev_des = None
        self._inverse_cache.clear()
