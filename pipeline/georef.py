"""Georeferencing: maps pixel coordinates to GPS using a homography transform."""

import json
from pathlib import Path

import cv2
import numpy as np


def load_reference(path: str) -> dict:
    """Load and validate a georef reference file.

    Expected format:
    {
      "reference_points": [
        {"pixel": [x, y], "gps": [lat, lon], "label": "optional description"},
        ...
      ]
    }

    Minimum 4 points required for homography.
    """
    data = json.loads(Path(path).read_text())

    points = data.get("reference_points", [])
    if len(points) < 4:
        raise ValueError(f"Need at least 4 reference points, got {len(points)}")

    for i, pt in enumerate(points):
        if "pixel" not in pt or "gps" not in pt:
            raise ValueError(f"Reference point {i} missing 'pixel' or 'gps' field")
        if len(pt["pixel"]) != 2 or len(pt["gps"]) != 2:
            raise ValueError(f"Reference point {i}: 'pixel' and 'gps' must be [x,y] and [lat,lon]")

    return data


def compute_homography(reference_points: list[dict]) -> np.ndarray:
    """Compute a 3x3 homography matrix mapping pixel coords to GPS coords.

    Args:
        reference_points: List of {"pixel": [x, y], "gps": [lat, lon]} dicts.

    Returns:
        3x3 numpy array (homography matrix).
    """
    src = np.array([pt["pixel"] for pt in reference_points], dtype=np.float64)
    dst = np.array([pt["gps"] for pt in reference_points], dtype=np.float64)

    # Use RANSAC if >4 points for robustness against outliers
    if len(reference_points) > 4:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        H, mask = cv2.findHomography(src, dst, 0)

    if H is None:
        raise ValueError("Could not compute homography — check that reference points are not collinear")

    return H


def pixel_to_gps(homography: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Transform a pixel coordinate to GPS using the homography.

    Args:
        homography: 3x3 homography matrix from compute_homography().
        x: Pixel x coordinate.
        y: Pixel y coordinate.

    Returns:
        (latitude, longitude) tuple.
    """
    pt = np.array([x, y, 1.0], dtype=np.float64)
    result = homography @ pt
    result /= result[2]  # normalize homogeneous coords
    return (float(result[0]), float(result[1]))


def bbox_to_gps(homography: np.ndarray, bbox: np.ndarray | list) -> tuple[float, float]:
    """Compute GPS for an entity's ground contact point (bottom-center of bbox).

    Args:
        homography: 3x3 homography matrix.
        bbox: [x1, y1, x2, y2] bounding box.

    Returns:
        (latitude, longitude) tuple.
    """
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) / 2.0  # horizontal center
    foot_y = y2                 # bottom edge (ground contact)
    return pixel_to_gps(homography, foot_x, foot_y)
