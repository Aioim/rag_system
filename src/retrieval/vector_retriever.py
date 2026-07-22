"""VectorRetriever — 查询编码 + FAISS 向量召回"""
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from retrieval.store import FAISSStore


class VectorRetriever:
    """查询 → encoder 编码（COSINE 时归一化，与写入侧一致）→ FAISS 搜索"""

    def __init__(self, store: "FAISSStore", encoder: "SentenceTransformer"):
        self._store = store
        self._encoder = encoder

    def retrieve(self, query: str, k: int) -> list[str]:
        from config import settings

        vec = np.asarray(self._encoder.encode([query]), dtype=np.float32)
        if settings.faiss.metric_type == "COSINE":
            faiss.normalize_L2(vec)
        return self._store.search(vec[0], k)
