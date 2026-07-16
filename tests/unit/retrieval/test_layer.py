"""RetrievalLayer 编排 + 单例测试（小索引端到端，mock 模型）"""
import typing

import numpy as np
import pytest

from models.context import PipelineContext
from models.enums import RetrievalEval
from retrieval import RetrievalLayer, get_retrieval_layer, reset_retrieval_layer
from tests.unit.retrieval.conftest import DIM, make_chunk, one_hot, write_chunks


class MockEncoder:
    """按关键词映射到 one-hot 向量"""

    TOPICS: typing.ClassVar[dict[str, int]] = {"年假": 0, "薪资": 1, "报销": 2}

    def encode(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(DIM, dtype=np.float32)
            for kw, i in self.TOPICS.items():
                if kw in t:
                    v[i] = 1.0
            vecs.append(v)
        return np.array(vecs)


class MockCrossEncoder:
    def predict(self, pairs):
        return np.array(
            [0.9 if "年假" in text else 0.1 for _, text in pairs]
        )


def _write_corpus():
    # d1: c0 <-> c1（年假文档两段，验证扩展）；d2/d3 干扰项
    write_chunks([
        make_chunk(0, "申请年假需提前三天提交审批", one_hot(0), next_id="c1"),
        make_chunk(1, "年假审批需附上假条材料", one_hot(0), prev_id="c0"),
        make_chunk(2, "薪资明细可在人事系统查询", one_hot(1), doc_id="d2"),
        make_chunk(3, "差旅报销需提供发票原件", one_hot(2), doc_id="d3"),
    ])


def _layer() -> RetrievalLayer:
    return RetrievalLayer(encoder=MockEncoder(), cross_encoder=MockCrossEncoder())


class TestRetrievalLayer:
    async def test_end_to_end(self, faiss_env):
        _write_corpus()
        ctx = PipelineContext(query="申请年假需要什么材料？", collection="test")
        ctx.rewritten_queries = ["申请年假需要什么材料？", "年假 材料 审批"]
        ctx = await _layer().retrieve(ctx)

        assert ctx.candidates, "RRF 融合后应有候选"
        assert ctx.reranked
        # 年假相关 chunk 精排最高
        assert "年假" in ctx.reranked[0].text
        assert ctx.reranked[0].rerank_score == pytest.approx(0.9)
        # 上下文扩展：c0 的窗口应包含 c1 文本
        top = ctx.reranked[0]
        assert len(top.metadata["window_chunk_ids"]) >= 2
        # Self-RAG 自评已写入
        assert ctx.retrieval_eval is not None
        # 耗时埋点
        assert "retrieval_recall_ms" in ctx.metadata
        assert "retrieval_rerank_ms" in ctx.metadata

    async def test_no_rewritten_queries_falls_back_to_query(self, faiss_env):
        _write_corpus()
        ctx = PipelineContext(query="年假审批", collection="test")
        ctx = await _layer().retrieve(ctx)
        assert ctx.reranked

    async def test_missing_collection_raises(self, faiss_env):
        ctx = PipelineContext(query="年假", collection="ghost")
        with pytest.raises(ValueError, match="ghost"):
            await _layer().retrieve(ctx)

    async def test_reranked_truncated_to_top_k(self, faiss_env):
        from config import settings

        _write_corpus()
        saved = settings.retrieval.top_k
        settings.retrieval.top_k = 2
        try:
            ctx = PipelineContext(query="年假 薪资 报销", collection="test")
            ctx = await _layer().retrieve(ctx)
            assert len(ctx.reranked) <= 2
        finally:
            settings.retrieval.top_k = saved

    async def test_empty_collection_returns_insufficient(self, faiss_env):
        """目录存在但无索引文件 → 空结果 + INSUFFICIENT，不报错"""
        (faiss_env / "empty").mkdir()
        ctx = PipelineContext(query="年假", collection="empty")
        ctx = await _layer().retrieve(ctx)
        assert ctx.candidates == []
        assert ctx.reranked == []
        assert ctx.retrieval_eval == RetrievalEval.INSUFFICIENT

    async def test_single_path_failure_degrades(self, faiss_env, monkeypatch):
        """BM25 路异常时向量路仍可用（单路降级）"""
        _write_corpus()
        from retrieval.bm25_retriever import BM25Retriever

        def _boom(self, query, k):
            raise RuntimeError("boom")

        monkeypatch.setattr(BM25Retriever, "retrieve", _boom)
        ctx = PipelineContext(query="年假审批", collection="test")
        ctx = await _layer().retrieve(ctx)
        assert ctx.reranked, "向量单路仍应产出结果"


class TestSingleton:
    def test_get_returns_same_instance(self):
        reset_retrieval_layer()
        try:
            assert get_retrieval_layer() is get_retrieval_layer()
        finally:
            reset_retrieval_layer()

    def test_reset_creates_new_instance(self):
        reset_retrieval_layer()
        a = get_retrieval_layer()
        reset_retrieval_layer()
        assert get_retrieval_layer() is not a
        reset_retrieval_layer()
