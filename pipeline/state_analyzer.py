"""State analyzer: uses Gemma 4 to describe the state of cropped entities.

Improvement: analyze_entities_batch() now composites all entity crops into a
numbered grid image and makes a single Gemma call, rather than one call per
entity.  This amortizes vision-backbone overhead and reduces total inference
time when tracking many entities per frame.
"""

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scene_analyzer import SceneAnalyzer

STATE_PROMPT_SINGLE = (
    "Describe this {label}'s current state as a short comma-separated list of attributes. "
    "Include: action/pose (e.g., running, sitting, parked, moving), "
    "spatial context (e.g., near building, center of road, on sidewalk), "
    "and any notable visual attributes (e.g., wearing red, large, partially occluded). "
    "Return ONLY the comma-separated list, nothing else. Keep each item under 5 words."
)

GRID_PROMPT = (
    "The image shows a numbered grid of cropped entities. "
    "For each numbered cell (1, 2, 3, …), describe that entity's state as a short "
    "comma-separated list: action/pose, spatial context, notable visual attributes. "
    "Reply with one line per entity in the format:\n"
    "1: running, center of field, wearing red\n"
    "2: standing, near goalpost, yellow jersey\n"
    "Keep each attribute under 5 words. Only output the numbered lines, nothing else."
)

# Crop size for each cell in the batch grid
CELL_W = 128
CELL_H = 192
GRID_COLS = 4  # max columns in the composite grid


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
        """Crop an entity from the frame and analyze its state (single call).

        Args:
            frame: Full BGR frame as numpy array.
            bbox: [x1, y1, x2, y2] bounding box.
            label: Entity label (e.g., "person").
            padding: Fraction of box size to add as context padding.

        Returns:
            List of state strings (e.g., ["running", "near goalpost"]).
        """
        crop = _padded_crop(frame, bbox, padding)
        if crop is None:
            return ["unknown"]

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        self.load()
        prompt = STATE_PROMPT_SINGLE.format(label=label)
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
        """Analyze states for multiple entities using a single Gemma call.

        All entity crops are composited into a numbered grid image and sent
        to Gemma in one request.  This reduces model overhead when many
        entities are present.

        Args:
            frame: Full BGR frame.
            boxes: (N, 4) bounding boxes.
            labels: N labels.
            max_entities: Cap on how many entities to analyze per frame.

        Returns:
            List of state lists, one per entity (input order preserved).
        """
        n_total = len(labels)
        n = min(n_total, max_entities)

        if n == 0:
            return []

        # Build list of crops (None if extraction fails)
        crops: list[np.ndarray | None] = []
        for i in range(n):
            crop = _padded_crop(frame, boxes[i], padding=0.15)
            crops.append(crop)

        # If only one entity, fall back to single-call path (no grid overhead)
        if n == 1:
            if crops[0] is not None:
                states = self.analyze_entity(frame, boxes[0], labels[0])
            else:
                states = ["unknown"]
            result = [states]
            result += [["not analyzed"]] * (n_total - n)
            return result

        # Build numbered grid image
        grid_pil = _build_grid(crops, cell_w=CELL_W, cell_h=CELL_H, cols=GRID_COLS)

        self.load()
        raw = self.analyzer._generate_raw(grid_pil, GRID_PROMPT)

        # Parse "N: attr1, attr2, ..." lines
        parsed: dict[int, list[str]] = {}
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            num_part, _, rest = line.partition(":")
            try:
                idx = int(num_part.strip())
            except ValueError:
                continue
            states = [s.strip().lower() for s in rest.split(",")]
            states = [s for s in states if s and len(s) < 50]
            parsed[idx] = states if states else ["unknown"]

        # Assemble results in order; fall back to single-call if Gemma missed an entity
        results: list[list[str]] = []
        for i in range(n):
            cell_num = i + 1
            if cell_num in parsed:
                results.append(parsed[cell_num])
            else:
                # Gemma missed this cell — fall back to individual crop analysis
                if crops[i] is not None:
                    states = self.analyze_entity(frame, boxes[i], labels[i])
                else:
                    states = ["unknown"]
                results.append(states)

        # Entities beyond the cap
        results += [["not analyzed"]] * (n_total - n)
        return results


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _padded_crop(
    frame: np.ndarray, bbox: np.ndarray, padding: float = 0.15
) -> np.ndarray | None:
    """Extract a padded crop of an entity from a BGR frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _build_grid(
    crops: list[np.ndarray | None],
    cell_w: int = CELL_W,
    cell_h: int = CELL_H,
    cols: int = GRID_COLS,
) -> Image.Image:
    """Composite crops into a numbered grid PIL image.

    Each cell is resized to (cell_w, cell_h). Empty crops are filled with grey.
    A number label is drawn in the top-left corner of each cell.
    """
    n = len(crops)
    rows = math.ceil(n / cols)
    grid_w = cols * cell_w
    grid_h = rows * cell_h

    grid = Image.new("RGB", (grid_w, grid_h), color=(40, 40, 40))
    draw = ImageDraw.Draw(grid)

    for i, crop in enumerate(crops):
        row, col = divmod(i, cols)
        x_off = col * cell_w
        y_off = row * cell_h

        if crop is not None:
            # Convert BGR → RGB and resize
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_cell = Image.fromarray(rgb).resize((cell_w, cell_h), Image.LANCZOS)
        else:
            pil_cell = Image.new("RGB", (cell_w, cell_h), color=(80, 80, 80))

        grid.paste(pil_cell, (x_off, y_off))

        # Draw cell number (white text, small shadow for readability)
        label = str(i + 1)
        draw.text((x_off + 4, y_off + 2), label, fill=(0, 0, 0))
        draw.text((x_off + 3, y_off + 1), label, fill=(255, 255, 255))

    return grid
