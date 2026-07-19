"""SupplementaryRetriever 测试"""
import pytest
from models.chunk import Chunk
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval
from fallback.supplementary import SupplementaryRetriever


def _make_chunk(chunk_id: str, text: str = "", score: float = 0.5) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text or f"text_{chunk_id}",
        chunk_index=0,
        rerank_score=score,
    )


class _FakeRetrievalLayer:
    """模拟检索层 — 接受 top_k 参数"""

    def __init__(self, chunk_count: int = 3, retrieval_eval=RetrievalEval.SUFFICIENT):
        self._chunk_count = chunk_count
        self._retrieval_eval = retrieval_eval
        self.retrieve_calls: list = []
        self.last_top_k: int | None = None

    async def retrieve(self, ctx, top_k=None):
        self.retrieve_calls.append(ctx.query)
        self.last_top_k = top_k
        ctx.reranked = [
            _make_chunk(f"new_{i}", score=0.7) for i in range(self._chunk_count)
        ]
        ctx.retrieval_eval = self._retrieval_eval
        return ctx


class TestSupplementaryRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_need_more_supplements_results(self):
        """NEED_MORE 时补充检索成功，合并新旧结果"""
        retriever = SupplementaryRetriever()
        retrieval_layer = _FakeRetrievalLayer(
            chunk_count=4, retrieval_eval=RetrievalEval.SUFFICIENT
        )

        ctx = PipelineContext(query="测试问题")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = [
            _make_chunk("old_0", score=0.4),
            _make_chunk("old_1", score=0.3),
        ]

        ctx = await retriever.retrieve(ctx, retrieval_layer)

        assert len(ctx.reranked) >= 4
        new_ids = {c.chunk_id for c in ctx.reranked}
        assert "old_0" in new_ids
        assert "old_1" in new_ids
        # 验证 top_k 被放宽传递
        assert retrieval_layer.last_top_k is not None

    @pytest.mark.asyncio
    async def test_retrieve_dedup_merges_correctly(self):
        """去重合并：新旧有重叠时正确合并"""

        class OverlapRetrievalLayer:
            async def retrieve(self, ctx, top_k=None):
                ctx.reranked = [
                    _make_chunk("old_0", score=0.7),
                    _make_chunk("new_1", score=0.6),
                ]
                ctx.retrieval_eval = RetrievalEval.NEED_MORE
                return ctx

        retriever = SupplementaryRetriever()
        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = [
            _make_chunk("old_0", score=0.5),
            _make_chunk("old_1", score=0.4),
        ]

        ctx = await retriever.retrieve(ctx, OverlapRetrievalLayer())

        ids = {c.chunk_id for c in ctx.reranked}
        assert len(ids) == 3
        assert "old_1" in ids

    @pytest.mark.asyncio
    async def test_retrieve_marks_fallback_partial(self):
        """补充检索后若仍 NEED_MORE，标记 fallback_level = PARTIAL"""
        retriever = SupplementaryRetriever()
        retrieval_layer = _FakeRetrievalLayer(
            chunk_count=3, retrieval_eval=RetrievalEval.NEED_MORE
        )

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = []

        ctx = await retriever.retrieve(ctx, retrieval_layer)

        assert ctx.fallback_level == FallbackLevel.PARTIAL
        assert ctx.is_fallback is True

    @pytest.mark.asyncio
    async def test_retrieve_exception_preserves_original(self):
        """补充检索异常时保留原始结果"""

        class FailingRetrievalLayer:
            async def retrieve(self, ctx, top_k=None):
                raise RuntimeError("检索服务不可用")

        retriever = SupplementaryRetriever()
        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = [_make_chunk("old_0")]

        ctx = await retriever.retrieve(ctx, FailingRetrievalLayer())

        assert len(ctx.reranked) == 1
        assert ctx.reranked[0].chunk_id == "old_0"
        assert ctx.fallback_level == FallbackLevel.PARTIAL
        assert ctx.retrieval_eval == RetrievalEval.NEED_MORE

    @pytest.mark.asyncio
    async def test_top_k_relaxed_via_parameter(self):
        """补充检索通过参数传递放宽的 top_k，不修改全局配置"""
        from config import settings

        original_top_k = settings.retrieval.top_k

        retriever = SupplementaryRetriever()
        retrieval_layer = _FakeRetrievalLayer()

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = []

        await retriever.retrieve(ctx, retrieval_layer)

        # 全局配置未被修改
        assert settings.retrieval.top_k == original_top_k
        # 检索层收到了放宽后的 top_k
        assert retrieval_layer.last_top_k == min(
            original_top_k * SupplementaryRetriever.TOP_K_MULTIPLIER,
            SupplementaryRetriever.TOP_K_MAX,
        )


class TestMergedReevaluation:
    """审查 H14：合并旧结果改变了 reranked 后，评估必须重新计算以反映合并后数据"""

    @pytest.mark.asyncio
    async def test_low_score_old_chunks_downgrade_optimistic_eval(self):
        """新结果单独评估为 SUFFICIENT，但合并低分旧结果后均分低于阈值，
        评估不应停留在乐观的 SUFFICIENT"""

        class SufficientNewLayer:
            async def retrieve(self, ctx, top_k=None):
                ctx.reranked = [_make_chunk("new_0", score=0.55)]
                ctx.retrieval_eval = RetrievalEval.SUFFICIENT  # 仅反映 new_0
                return ctx

        retriever = SupplementaryRetriever()
        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = [
            _make_chunk("old_0", score=0.1),
            _make_chunk("old_1", score=0.1),
        ]

        ctx = await retriever.retrieve(ctx, SufficientNewLayer())

        # 合并后 avg=(0.55+0.1+0.1)/3=0.25 < sufficient 阈值(0.5)
        # 重评估为 INSUFFICIENT 后由"旧结果存在"守卫恢复到 NEED_MORE
        assert ctx.retrieval_eval == RetrievalEval.NEED_MORE

    @pytest.mark.asyncio
    async def test_high_score_old_chunks_upgrade_eval(self):
        """合并高分旧结果后均分达到 SUFFICIENT，评估应同步上调"""

        class NeedMoreNewLayer:
            async def retrieve(self, ctx, top_k=None):
                ctx.reranked = [_make_chunk("new_0", score=0.4)]
                ctx.retrieval_eval = RetrievalEval.NEED_MORE
                return ctx

        retriever = SupplementaryRetriever()
        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = [
            _make_chunk("old_0", score=0.8),
            _make_chunk("old_1", score=0.8),
        ]

        ctx = await retriever.retrieve(ctx, NeedMoreNewLayer())

        # 合并后 avg=(0.4+0.8+0.8)/3=0.667 >= 0.5 → SUFFICIENT，不再标 PARTIAL
        assert ctx.retrieval_eval == RetrievalEval.SUFFICIENT
        assert ctx.is_fallback is False
