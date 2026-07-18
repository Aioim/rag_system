"""Reranker — CrossEncoder 精排 + MMR 多样性选择

BGE reranker num_labels=1，sentence-transformers CrossEncoder.predict
默认经 Sigmoid 激活输出 0~1，与 relevance_threshold_* 阈值同量纲。
MMR 多样性用原始 chunk 向量（FAISS reconstruct）计算余弦——窗口扩展文本
无现成 embedding，原始向量是足够好的近似（设计文档 5.6 已评审确认）。
"""
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from models.chunk import Chunk

_cross_encoder: "CrossEncoder | None" = None
_ce_lock = threading.Lock()


def load_cross_encoder() -> "CrossEncoder":
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
    def __init__(self, cross_encoder: "CrossEncoder"):
        self._ce = cross_encoder

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """(query, chunk.text) 逐对打分写入 rerank_score，按分数降序返回"""
        if not chunks:
            return []
        results = self._ce.rank(query, [c.text for c in chunks])
        ranked: list[Chunk] = []
        for r in results:
            c = chunks[r["corpus_id"]]
            c.rerank_score = float(r["score"])
            ranked.append(c)
        return ranked


def _normalize(v: np.ndarray | None) -> np.ndarray | None:
    """单位化向量；None 或零向量返回 None（余弦按 0 处理）"""
    if v is None:
        return None
    norm = float(np.linalg.norm(v))
    if norm < 1e-10:
        return None
    return v / norm


def mmr_select(
    chunks: list[Chunk],
    vectors: dict[str, np.ndarray | None],
    top_k: int,
    mmr_lambda: float,
) -> list[Chunk]:
    """MMR 贪心：score = λ·rerank_score - (1-λ)·max_sim(已选)

    输入契约：chunks 须已按 rerank_score 降序（Reranker.rerank 的输出）。
    向量缺失（reconstruct 失败）相似度按 0 处理，即视为完全多样。
    """
    if not chunks:
        return []
    if len(chunks) <= top_k or top_k <= 0:
        return list(chunks[: max(top_k, 0)])

    # 预归一化：余弦退化为点积，避免循环中重复计算范数
    unit_vecs = {cid: _normalize(v) for cid, v in vectors.items()}

    def _sim(a_id: str, b_id: str) -> float:
        a, b = unit_vecs.get(a_id), unit_vecs.get(b_id)
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    pool = list(chunks)
    selected = [pool.pop(0)]
    # 每个候选与已选集合的最大相似度，增量维护避免每轮全量重算
    max_sims = [0.0] * len(pool)
    last = selected[0]
    while pool and len(selected) < top_k:
        best_idx, best_score = 0, float("-inf")
        for i, c in enumerate(pool):
            max_sims[i] = max(max_sims[i], _sim(c.chunk_id, last.chunk_id))
            score = mmr_lambda * c.rerank_score - (1 - mmr_lambda) * max_sims[i]
            if score > best_score:
                best_idx, best_score = i, score
        last = pool.pop(best_idx)
        max_sims.pop(best_idx)
        selected.append(last)
    return selected
