"""RetrievalEvaluator — Self-RAG 自评：top_k 平均 rerank_score 对照阈值"""
from models.chunk import Chunk
from models.enums import RetrievalEval


def evaluate(reranked: list[Chunk]) -> RetrievalEval:
    """avg(rerank_score) >= 0.5 → SUFFICIENT；>= 0.3 → NEED_MORE；否则 INSUFFICIENT"""
    if not reranked:
        return RetrievalEval.INSUFFICIENT

    from config import settings

    cfg = settings.retrieval
    # 防御性断言：配置错误时尽早暴露，避免自评逻辑反转
    assert cfg.relevance_threshold_sufficient >= cfg.relevance_threshold_need_more, (
        f"配置错误: relevance_threshold_sufficient ({cfg.relevance_threshold_sufficient}) "
        f"必须 >= relevance_threshold_need_more ({cfg.relevance_threshold_need_more})"
    )
    avg = sum(c.rerank_score for c in reranked) / len(reranked)
    if avg >= cfg.relevance_threshold_sufficient:
        return RetrievalEval.SUFFICIENT
    if avg >= cfg.relevance_threshold_need_more:
        return RetrievalEval.NEED_MORE
    return RetrievalEval.INSUFFICIENT
