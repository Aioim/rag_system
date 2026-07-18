"""RetrievalEvaluator — Self-RAG 自评：top_k 平均 rerank_score 对照阈值"""
from models.chunk import Chunk
from models.enums import RetrievalEval


def evaluate(reranked: list[Chunk]) -> RetrievalEval:
    """avg(rerank_score) >= 0.5 → SUFFICIENT；>= 0.3 → NEED_MORE；否则 INSUFFICIENT"""
    if not reranked:
        return RetrievalEval.INSUFFICIENT

    from config import settings

    cfg = settings.retrieval
    # 阈值顺序由 RetrievalConfig 的 model_validator 在配置加载时保证
    avg = sum(c.rerank_score for c in reranked) / len(reranked)
    if avg >= cfg.relevance_threshold_sufficient:
        return RetrievalEval.SUFFICIENT
    if avg >= cfg.relevance_threshold_need_more:
        return RetrievalEval.NEED_MORE
    return RetrievalEval.INSUFFICIENT
