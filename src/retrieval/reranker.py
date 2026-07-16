"""Reranker — CrossEncoder 精排 + MMR 多样性选择

BGE reranker num_labels=1，sentence-transformers CrossEncoder.predict
默认经 Sigmoid 激活输出 0~1，与 relevance_threshold_* 阈值同量纲。
MMR 多样性用原始 chunk 向量（FAISS reconstruct）计算余弦——窗口扩展文本
无现成 embedding，原始向量是足够好的近似（设计文档 5.6 已评审确认）。
"""
import threading

import numpy as np

from models.chunk import Chunk

_cross_encoder = None
_ce_lock = threading.Lock()


def load_cross_encoder():
    """从本地路径加载 CrossEncoder（进程内缓存）；未下载抛 RuntimeError"""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    with _ce_lock:
        if _cross_encoder is None:
            from config import settings
            from model import models

            path = models.get_path("rerank")
            if path is None:
                raise RuntimeError(
                    "Rerank 模型未下载，请先执行 "
                    "`from model import models; models.download('rerank')`"
                )
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder(
                str(path), device=settings.embedding.device
            )
    return _cross_encoder


class Reranker:
    def __init__(self, cross_encoder):
        self._ce = cross_encoder

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """(query, chunk.text) 逐对打分写入 rerank_score，按分数降序返回"""
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._ce.predict(pairs)
        for c, s in zip(chunks, scores, strict=True):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def mmr_select(
    chunks: list[Chunk],
    vectors: dict[str, np.ndarray | None],
    top_k: int,
    mmr_lambda: float,
) -> list[Chunk]:
    """MMR 贪心：score = λ·rerank_score - (1-λ)·max_sim(已选)"""
    if not chunks:
        return []
    # 按 rerank_score 降序排序（调用方传入已排序列表，此为防御性保证）
    pool = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    if len(pool) <= top_k or top_k <= 0:
        return pool[:max(top_k, 0)]

    selected = [pool.pop(0)]
    while pool and len(selected) < top_k:
        best_idx, best_score = 0, float("-inf")
        for i, c in enumerate(pool):
            max_sim = max(
                _cosine(vectors.get(c.chunk_id), vectors.get(s.chunk_id))
                for s in selected
            )
            score = mmr_lambda * c.rerank_score - (1 - mmr_lambda) * max_sim
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(pool.pop(best_idx))
    return selected
