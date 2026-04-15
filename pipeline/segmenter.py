"""SAM3 segmenter: detects and segments objects in images using text prompts."""

import gc
from dataclasses import dataclass

import cv2
import mlx.core as mx
import numpy as np
from mlx_vlm.models.sam3.generate import Sam3Predictor, predict_multi
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


DEFAULT_INFERENCE_WIDTH = 640


class Segmenter:
    def __init__(self, model_id: str = MODEL_ID, threshold: float = 0.3, inference_width: int = DEFAULT_INFERENCE_WIDTH):
        self.model_id = model_id
        self.threshold = threshold
        self.inference_width = inference_width
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

    def _downscale(self, image: Image.Image) -> tuple[Image.Image, float]:
        """Downscale image to inference_width, return (scaled_image, scale_factor).

        If image is already smaller than inference_width, return it unchanged.
        """
        orig_w, orig_h = image.size
        if orig_w <= self.inference_width:
            return image, 1.0

        scale = self.inference_width / orig_w
        new_w = self.inference_width
        new_h = int(orig_h * scale)
        return image.resize((new_w, new_h), Image.LANCZOS), scale

    def _upscale_results(
        self, boxes: np.ndarray, masks: np.ndarray, scale: float, orig_w: int, orig_h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scale boxes and masks back to original resolution."""
        if scale == 1.0:
            return boxes, masks

        # Scale bounding boxes back up
        boxes = boxes / scale

        # Resize each mask to original dimensions
        upscaled = np.zeros((masks.shape[0], orig_h, orig_w), dtype=masks.dtype)
        for i in range(masks.shape[0]):
            upscaled[i] = cv2.resize(masks[i].astype(np.float32), (orig_w, orig_h)) > 0.5
        return boxes, upscaled

    @staticmethod
    def _boxes_from_masks(masks: np.ndarray, fallback_boxes: np.ndarray) -> np.ndarray:
        """Derive tight bounding boxes from binary masks.

        Falls back to the model's predicted box if a mask is empty.
        """
        boxes = np.empty_like(fallback_boxes)
        for i in range(len(masks)):
            ys, xs = np.where(masks[i] > 0)
            if len(xs) == 0:
                boxes[i] = fallback_boxes[i]
            else:
                boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
        return boxes

    def detect(self, image: Image.Image, labels: list[str]) -> Detection:
        """Run detection + segmentation for all labels in a single batched call.

        Uses predict_multi() to run the vision backbone once and reuse it
        across all label prompts. Downscales the image for faster inference,
        then scales results back to the original resolution.

        Args:
            image: PIL Image to analyze.
            labels: List of text prompts (e.g., ["person", "car"]).

        Returns:
            Detection with results at original image resolution.
        """
        self.load()

        orig_w, orig_h = image.size
        small_image, scale = self._downscale(image)
        if scale != 1.0:
            sw, sh = small_image.size
            print(f" [{orig_w}x{orig_h} → {sw}x{sh}]", end="", flush=True)

        # Single batched call: ViT runs once, text+DETR per label
        result = predict_multi(self.predictor, small_image, labels)

        n = len(result.scores)
        if n == 0:
            return Detection(
                boxes=np.empty((0, 4)),
                masks=np.empty((0, orig_h, orig_w)),
                scores=np.empty((0,)),
                labels=[],
            )

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

        # Scale back to original resolution
        boxes, masks = self._upscale_results(boxes, masks, scale, orig_w, orig_h)

        # Recompute boxes from masks — SAM3's box regression can misalign
        # when the image is squished to square for inference
        boxes = self._boxes_from_masks(masks, boxes)

        return Detection(
            boxes=boxes,
            masks=masks,
            scores=scores,
            labels=list(result.labels),
        )
