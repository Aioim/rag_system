"""VectorRetriever — 查询编码 + FAISS 向量召回"""
import threading
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from retrieval.store import FAISSStore

_embedding_model: "SentenceTransformer | None" = None
_model_lock = threading.Lock()


def load_embedding_model() -> "SentenceTransformer":
    """从本地路径加载 SentenceTransformer（进程内缓存）

    未下载时抛 RuntimeError 并提示下载命令，不自动触发下载。
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _model_lock:
        if _embedding_model is None:
            from config import settings
            from model import models

            path = models.get_path("embedding")
            if path is None:
                raise RuntimeError(
                    "Embedding 模型未下载，请先执行 "
                    "`from model import models; models.download('embedding')`"
                )
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer(
                str(path), device=settings.embedding.device
            )
    return _embedding_model


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
