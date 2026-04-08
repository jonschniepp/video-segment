"""SAM3 segmenter: detects and segments objects in images using text prompts."""

import gc
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.utils import get_model_path, load_model
from PIL import Image

MODEL_ID = "mlx-community/sam3-bf16"


@dataclass
class Detection:
    boxes: np.ndarray      # (N, 4) xyxy pixel coords
    masks: np.ndarray      # (N, H, W) binary masks
    scores: np.ndarray     # (N,) confidence scores
    labels: list[str]      # (N,) label per detection


class Segmenter:
    def __init__(self, model_id: str = MODEL_ID, threshold: float = 0.3):
        self.model_id = model_id
        self.threshold = threshold
        self.predictor = None

    def load(self):
        if self.predictor is None:
            print(f"Loading SAM3 from {self.model_id}...")
            model_path = get_model_path(self.model_id)
            model = load_model(model_path)
            processor = Sam3Processor.from_pretrained(str(model_path))
            self.predictor = Sam3Predictor(model, processor, score_threshold=self.threshold)
            print("SAM3 loaded.")

    def unload(self):
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
            gc.collect()
            mx.metal.clear_cache()
            print("SAM3 unloaded.")

    def detect(self, image: Image.Image, labels: list[str]) -> Detection:
        """Run detection + segmentation for each label.

        Args:
            image: PIL Image to analyze.
            labels: List of text prompts (e.g., ["person", "car"]).

        Returns:
            Detection with combined results across all labels.
        """
        self.load()

        all_boxes = []
        all_masks = []
        all_scores = []
        all_labels = []

        for label in labels:
            result = self.predictor.predict(image, text_prompt=label)

            n = len(result.scores)
            if n == 0:
                continue

            boxes = np.array(result.boxes[:n])
            scores = np.array(result.scores[:n])

            # Convert masks: may be mx.array, convert to numpy
            masks = result.masks[:n]
            if isinstance(masks, mx.array):
                masks = np.array(masks)
            elif not isinstance(masks, np.ndarray):
                masks = np.array(masks)

            # Ensure masks are 3D: (N, H, W)
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]

            all_boxes.append(boxes)
            all_masks.append(masks)
            all_scores.append(scores)
            all_labels.extend([label] * n)

        if not all_boxes:
            h, w = image.size[1], image.size[0]
            return Detection(
                boxes=np.empty((0, 4)),
                masks=np.empty((0, h, w)),
                scores=np.empty((0,)),
                labels=[],
            )

        return Detection(
            boxes=np.concatenate(all_boxes, axis=0),
            masks=np.concatenate(all_masks, axis=0),
            scores=np.concatenate(all_scores, axis=0),
            labels=all_labels,
        )
