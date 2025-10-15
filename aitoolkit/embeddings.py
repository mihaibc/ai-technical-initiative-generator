from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

_MODEL_ID = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_EMB = None


def _ensure_model():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer

        _EMB = SentenceTransformer(_MODEL_ID)
    return _EMB


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _ensure_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def top_k_similar(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 5) -> List[int]:
    sims = (doc_vecs @ query_vec).reshape(-1)
    idx = np.argsort(-sims)[:k]
    return idx.tolist()

