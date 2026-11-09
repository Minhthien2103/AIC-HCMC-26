import sys
import torch
import open_clip
import numpy as np
from pathlib import Path
from collections.abc import Sequence

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QueryEncoder():
    def __init__(self, model_name: str, pretrained: str, device: str = None):
        if device is None:
            self.device = DEVICE
        else:
            self.device = device

        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None

        self._load_model()


    def _load_model(self):
        print(f"Loading CLIP model: {self.model_name} ({self.pretrained})")

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name = self.model_name,
                pretrained = self.pretrained,
                device = self.device
            )

            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()

        except Exception as exc:
            raise RuntimeError(f"Unable to load CLIP {self.model_name} ({self.pretrained})") from exc

    @property
    def text_context_length(self) -> int:
        """Maximum token count accepted by the loaded CLIP text tower."""
        if self.model is not None and getattr(self.model, "context_length", None):
            return int(self.model.context_length)
        return int(getattr(self.tokenizer, "context_length", 77))

    def text_token_count(self, query: str) -> int:
        """Count tokens without CLIP's default 77-token truncation."""
        if self.tokenizer is None:
            raise RuntimeError("CLIP tokenizer is not initialized")
        # Query packs are short, but this generous context preserves the true
        # count while the caller decides how to split at the real context size.
        tokens = self.tokenizer([query], context_length=max(256, self.text_context_length))
        return int((tokens[0] != 0).sum().item())

    def _require_model(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CLIP model/tokenizer is not initialized")

    def encode_text_batch(self, queries: Sequence[str]) -> np.ndarray:
        """Encode a batch of textual KIS variants in one CUDA forward pass."""
        self._require_model()
        values = [str(query) for query in queries]
        if not values:
            return np.empty((0, 512), dtype=np.float32)

        tokens = self.tokenizer(values).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens).float()
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype(np.float32, copy=False)


    def encode_text(self, query: str) -> np.ndarray:
        return self.encode_text_batch([query])[0]


    def encode_image(self, image_input) -> np.ndarray:
        return self.encode_image_batch([image_input])[0]


    def encode_image_batch(self, image_inputs: Sequence) -> np.ndarray:
        """Encode a batch of RGB PIL images for resumable offline indexing."""
        self._require_model()
        if not image_inputs:
            return np.empty((0, 0), dtype=np.float32)
        img_tensor = torch.stack([self.preprocess(image) for image in image_inputs]).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(img_tensor).float()
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype(np.float32, copy=False)
