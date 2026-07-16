"""BM25Retriever — jieba 分词 + rank_bm25 内存稀疏索引

1K~10K 文档规模下启动时从 docstore 全量构建（秒级），不持久化。
"""
import jieba
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class BM25Retriever:
    """构建时记录 store.version，供上层判断索引热重载后是否需要重建"""

    def __init__(self, store):
        # 先读版本号再取数据：若 reload 在两者之间执行，version 为旧值
        # → 下次 _get_bm25 检测到版本不匹配 → 安全地多重建一次
        self.version = store.version
        pairs = store.all_chunks()
        self._chunk_ids = [cid for cid, _ in pairs]
        corpus = [_tokenize(text) for _, text in pairs]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(self, query: str, k: int) -> list[str]:
        """按 BM25 分数降序返回 chunk_id；score <= 0 的不返回"""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return [self._chunk_ids[i] for i in ranked[:k] if scores[i] > 0]
