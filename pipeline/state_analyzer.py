"""State analyzer: uses Gemma 4 to describe the state of cropped entities."""

import cv2
import numpy as np
from PIL import Image

from .scene_analyzer import SceneAnalyzer

STATE_PROMPT = (
    "Describe this {label}'s current state as a short comma-separated list of attributes. "
    "Include: action/pose (e.g., running, sitting, parked, moving), "
    "spatial context (e.g., near building, center of road, on sidewalk), "
    "and any notable visual attributes (e.g., wearing red, large, partially occluded). "
    "Return ONLY the comma-separated list, nothing else. Keep each item under 5 words."
)


class StateAnalyzer:
    """Analyzes cropped entity images with Gemma 4 to determine their states."""

    def __init__(self, analyzer: SceneAnalyzer | None = None):
        self.analyzer = analyzer or SceneAnalyzer()

    def load(self):
        self.analyzer.load()

    def unload(self):
        self.analyzer.unload()

    def analyze_entity(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        label: str,
        padding: float = 0.15,
    ) -> list[str]:
        """Crop an entity from the frame and analyze its state.

        Args:
            frame: Full BGR frame as numpy array.
            bbox: [x1, y1, x2, y2] bounding box.
            label: Entity label (e.g., "person").
            padding: Fraction of box size to add as context padding.

        Returns:
            List of state strings (e.g., ["running", "near goalpost"]).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        # Add padding for spatial context
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ["unknown"]

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # Use the scene analyzer's Gemma model with a state-specific prompt
        self.load()
        prompt = STATE_PROMPT.format(label=label)
        result = self.analyzer._generate_raw(pil_crop, prompt)

        states = [s.strip().lower() for s in result.split(",")]
        states = [s for s in states if s and len(s) < 50]
        return states if states else ["unknown"]

    def analyze_entities_batch(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        labels: list[str],
        max_entities: int = 10,
    ) -> list[list[str]]:
        """Analyze states for multiple entities in a frame.

        Args:
            frame: Full BGR frame.
            boxes: (N, 4) bounding boxes.
            labels: N labels.
            max_entities: Cap on how many entities to analyze per frame (for speed).

        Returns:
            List of state lists, one per entity.
        """
        results = []
        n = min(len(labels), max_entities)
        for i in range(n):
            states = self.analyze_entity(frame, boxes[i], labels[i])
            results.append(states)

        # Fill remaining with empty if we hit the cap
        for _ in range(len(labels) - n):
            results.append(["not analyzed"])

        return results
