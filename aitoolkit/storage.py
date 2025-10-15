from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "data")
CONNECTIONS_PATH = os.path.join(DATA_DIR, "connections.json")
INDEX_META_PATH = os.path.join(DATA_DIR, "index_meta.json")
INDEX_VECS_PATH = os.path.join(DATA_DIR, "index_vectors.npy")


@dataclass
class Document:
    text: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_connections() -> List[Dict[str, Any]]:
    ensure_data_dir()
    if not os.path.exists(CONNECTIONS_PATH):
        return []
    with open(CONNECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_connections(conns: List[Dict[str, Any]]):
    ensure_data_dir()
    with open(CONNECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(conns, f, indent=2)


def append_to_index(docs: List[Document], vectors: np.ndarray):
    ensure_data_dir()
    # Append vectors
    if os.path.exists(INDEX_VECS_PATH):
        old = np.load(INDEX_VECS_PATH)
        all_vecs = np.vstack([old, vectors])
    else:
        all_vecs = vectors
    np.save(INDEX_VECS_PATH, all_vecs)

    # Append meta
    meta = []
    if os.path.exists(INDEX_META_PATH):
        with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.extend([asdict(d) for d in docs])
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_index():
    if not os.path.exists(INDEX_VECS_PATH) or not os.path.exists(INDEX_META_PATH):
        return None, []
    vecs = np.load(INDEX_VECS_PATH)
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return vecs, meta

