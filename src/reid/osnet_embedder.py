"""Pretrained OSNet person appearance embeddings."""

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


class OSNetEmbedder:
    """Run OSNet x0.25 on batches of BGR person crops.

    The supplied Market-1501 checkpoint produces 512-dimensional appearance
    vectors from 256x128 RGB crops. Outputs are L2-normalized so their dot
    product is cosine similarity.
    """

    input_height = 256
    input_width = 128
    embedding_dimension = 512
    _mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    _std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        model_name: str = "osnet_x0_25",
        num_classes: int = 751,
        batch_size: int = 16,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if model_name != "osnet_x0_25":
            raise ValueError("This baseline currently supports model_name 'osnet_x0_25'")

        try:
            import torch
            from torchreid.models import build_model
            from torchreid.utils import load_pretrained_weights
        except Exception as exc:
            raise RuntimeError(
                "Could not import the OSNet Re-ID runtime. Install project dependencies."
            ) from exc

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Re-ID checkpoint was not found: {path}")
        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(batch_size)
        try:
            self.model = build_model(
                name=model_name,
                num_classes=int(num_classes),
                pretrained=False,
            )
            load_pretrained_weights(self.model, str(path))
            self.model.to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(f"Could not load OSNet checkpoint {path}: {exc}") from exc

    def embed(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Return one normalized 512-D feature vector for each valid BGR crop."""
        if not crops:
            return []
        tensors = [self._preprocess(crop) for crop in crops]
        embeddings = []
        with self._torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                batch = self._torch.stack(tensors[start : start + self.batch_size]).to(self.device)
                features = self.model(batch)
                features = self._torch.nn.functional.normalize(features, p=2, dim=1)
                embeddings.extend(features.cpu().numpy().astype(np.float32))
        return embeddings

    def _preprocess(self, crop: np.ndarray):
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            raise ValueError("OSNetEmbedder requires non-empty image crops")
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("OSNetEmbedder expects a three-channel BGR crop")
        resized = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - self._mean) / self._std
        chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        return self._torch.from_numpy(chw)
