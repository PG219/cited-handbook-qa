import os
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

class Embed:
    def __init__(self, provider: str = "local", model_name: str = "all-MiniLM-L6-v2"):
        self.provider = provider
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        ).astype("float32")
        norms = np.linalg.norm(embeddings, axis = 1, keepdims=True)
        embeddings =  embeddings/norms
        return embeddings
embedder = Embed(provider="local")
vectors = embedder.embed(["Hello world", "Machine Learning is Facinating."])
print(f"Vectors shape: {vectors.shape}")
