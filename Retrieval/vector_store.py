import faiss
import numpy as np
from typing import List, Tuple

class FaissIndex:
    def __init__(self, dim: int):
        # L2 distance index (simple & inspectable)
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray):
        """
        vectors: shape (N, D), dtype float32
        """
        assert vectors.dtype == np.float32, "FAISS requires float32"
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_vec: shape (1, D)
        returns: (distances, indices)
        """
        assert query_vec.ndim == 2 and query_vec.shape[0] == 1
        distances, indices = self.index.search(query_vec, k)
        return distances, indices
