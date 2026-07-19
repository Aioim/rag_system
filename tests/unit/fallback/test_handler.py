"""FallbackHandler 测试"""
import pytest
from models.chunk import Chunk
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval
from fallback.handler import FallbackHandler
from fallback.web_search import WebSearcher
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
    """模拟检索层 — 接受 top_k 参数

    score 需与 retrieval_eval 一致（补充检索合并后会按真实分数重评估）：
    NEED_MORE 场景应传 [0.3, 0.5) 区间的分数。
    """

    def __init__(
        self,
        chunk_count: int = 3,
        retrieval_eval=RetrievalEval.SUFFICIENT,
        score: float = 0.7,
    ):
        self._chunk_count = chunk_count
        self._retrieval_eval = retrieval_eval
        self._score = score
        self.last_top_k: int | None = None

    async def retrieve(self, ctx, top_k=None):
        self.last_top_k = top_k
        ctx.reranked = [
            _make_chunk(f"new_{i}", score=self._score)
            for i in range(self._chunk_count)
        ]
        ctx.retrieval_eval = self._retrieval_eval
        return ctx


@pytest.fixture
def handler():
    """创建测试用 FallbackHandler"""
    return FallbackHandler(
        web_searcher=WebSearcher(),
        supplementary=SupplementaryRetriever(),
    )


class TestFallbackHandler:
    # ---- NEED_MORE --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_need_more_triggers_supplementary_retrieval(self, handler):
        """NEED_MORE + retrieval_layer → 触发补充检索"""
        retrieval_layer = _FakeRetrievalLayer(
            chunk_count=5, retrieval_eval=RetrievalEval.NEED_MORE, score=0.4
        )

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = [_make_chunk("old_0")]

        ctx = await handler.handle(ctx, retrieval_layer)

        assert ctx.fallback_level == FallbackLevel.PARTIAL
        assert ctx.is_fallback is True
        assert len(ctx.reranked) >= 5
        # 验证 top_k 被放宽传递
        assert retrieval_layer.last_top_k is not None

    @pytest.mark.asyncio
    async def test_need_more_without_retrieval_layer_marks_partial(self, handler):
        """NEED_MORE 但无 retrieval_layer → 仅标记 PARTIAL"""
        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = [_make_chunk("old_0")]

        ctx = await handler.handle(ctx, retrieval_layer=None)

        assert ctx.fallback_level == FallbackLevel.PARTIAL
        assert ctx.is_fallback is True
        assert len(ctx.reranked) == 1  # 未改变

    # ---- INSUFFICIENT -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_insufficient_web_search_disabled_returns_no_answer(self, monkeypatch, handler):
        """INSUFFICIENT + 联网搜索禁用 → 诚实告知"""
        monkeypatch.setattr(
            "fallback.web_search.settings.web_search.enabled", False
        )
        ctx = PipelineContext(query="测试问题")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.fallback_level == FallbackLevel.NO_ANSWER
        assert result.is_fallback is True
        assert result.confidence == 0.0
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_insufficient_web_search_success(self, monkeypatch, handler):
        """INSUFFICIENT + 联网搜索成功 → WEB_SEARCH"""
        monkeypatch.setattr(
            "fallback.web_search.settings.web_search.enabled", True
        )

        async def mock_search(query):
            return "从网络获取的答案"

        handler._web_searcher.search = mock_search

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.fallback_level == FallbackLevel.WEB_SEARCH
        assert result.is_fallback is True
        assert result.answer == "从网络获取的答案"

    @pytest.mark.asyncio
    async def test_insufficient_web_search_fails_returns_no_answer(self, monkeypatch, handler):
        """INSUFFICIENT + 联网搜索失败 → 诚实告知"""
        monkeypatch.setattr(
            "fallback.web_search.settings.web_search.enabled", True
        )

        async def failing_search(query):
            return None

        handler._web_searcher.search = failing_search

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.fallback_level == FallbackLevel.NO_ANSWER
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_insufficient_web_search_exception_falls_back(self, monkeypatch, handler):
        """INSUFFICIENT + 联网搜索异常 → 诚实告知"""
        monkeypatch.setattr(
            "fallback.web_search.settings.web_search.enabled", True
        )

        async def error_search(query):
            raise RuntimeError("搜索服务不可用")

        handler._web_searcher.search = error_search

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.fallback_level == FallbackLevel.NO_ANSWER

    # ---- SUFFICIENT (不受影响) --------------------------------------------

    @pytest.mark.asyncio
    async def test_sufficient_passes_through(self, handler):
        """SUFFICIENT 时不做处理，直接返回 ctx"""
        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.SUFFICIENT
        ctx.answer = "已有答案"

        result = await handler.handle(ctx)
        assert result is ctx
        assert result.answer == "已有答案"
        assert result.fallback_level == FallbackLevel.NONE

    # ---- 构造参数注入 ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_custom_web_searcher_is_used(self):
        """自定义 WebSearcher 被正确使用"""

        class MockSearcher:
            async def search(self, query):
                return "Mock 结果"

        handler = FallbackHandler(
            web_searcher=MockSearcher(),
            supplementary=SupplementaryRetriever(),
        )
        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.answer == "Mock 结果"
        assert result.fallback_level == FallbackLevel.WEB_SEARCH

    @pytest.mark.asyncio
    async def test_custom_supplementary_is_used(self):
        """自定义 SupplementaryRetriever 被正确使用"""
        supp = SupplementaryRetriever()
        handler = FallbackHandler(
            web_searcher=WebSearcher(),
            supplementary=supp,
        )

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.reranked = [_make_chunk("old_0")]

        retrieval_layer = _FakeRetrievalLayer(
            chunk_count=3, retrieval_eval=RetrievalEval.NEED_MORE, score=0.4
        )
        result = await handler.handle(ctx, retrieval_layer)

        assert result.fallback_level == FallbackLevel.PARTIAL
        assert result.is_fallback is True
