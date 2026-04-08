"""Gemma 4 scene analyzer: analyzes video frames and generates object prompts for SAM3."""

import gc

import mlx.core as mx
from mlx_vlm import generate, load
from PIL import Image


SYSTEM_PROMPT = (
    "You are a vision system that identifies objects in images. "
    "When asked, list the distinct object types visible in the scene as simple, "
    "comma-separated labels suitable for an object detection model. "
    "Use singular nouns (e.g., 'person' not 'people'). "
    "Only list objects that are clearly visible. Be concise — labels only, no explanation."
)

USER_PROMPT_AUTO = "List every distinct object type visible in this image as comma-separated labels."

USER_PROMPT_QUERY = (
    "The user wants to find: '{query}'. "
    "Return a single, simple label suitable for an object detector. "
    "For example, if the user says 'find all people', return 'person'. "
    "Return only the label, nothing else."
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-8bit"


class SceneAnalyzer:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self):
        if self.model is None:
            print(f"Loading Gemma 4 from {self.model_id}...")
            self.model, self.processor = load(self.model_id)
            print("Gemma 4 loaded.")

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            mx.metal.clear_cache()
            print("Gemma 4 unloaded.")

    def analyze(self, image: Image.Image, query: str | None = None) -> list[str]:
        """Analyze a frame and return object labels for SAM3.

        Args:
            image: PIL Image of a video frame.
            query: Optional user query (e.g., "find all people"). If None, auto-detects objects.

        Returns:
            List of object label strings (e.g., ["person", "car", "bicycle"]).
        """
        self.load()

        if query:
            user_msg = USER_PROMPT_QUERY.format(query=query)
        else:
            user_msg = USER_PROMPT_AUTO

        prompt = self.processor.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_msg}]},
            ],
            add_generation_prompt=True,
        )

        response = generate(
            self.model,
            self.processor,
            prompt=prompt,
            image=image,
            max_tokens=100,
            temperature=0.1,
        )

        labels = [label.strip().lower() for label in response.strip().split(",")]
        labels = [l for l in labels if l and len(l) < 50]
        return labels
