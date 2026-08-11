import sys
import torch
import open_clip
import numpy as np
from pathlib import Path

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

        except Exception as e:
            print(f"    Loading model Error: {e}")


    def encode_text(self, query: str) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            print(f"    Model/Tokenizer is None")
            return np.zeros(512)

        tokens = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens).float()

        features = features.cpu().numpy()
        features /= np.linalg.norm(features, axis = 1, keepdims = True)

        return features[0]


    def encode_image(self, image_input) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            print(f"    Model/Tokenizer is None")
            return np.zeros(512)

        img_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(img_tensor).float()

        features = features.cpu().numpy()
        features /= np.linalg.norm(features, axis = 1, keepdims = True)

        return features[0]
