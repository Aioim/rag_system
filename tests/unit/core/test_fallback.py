"""FallbackHandler 测试"""
import pytest
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval
from core.fallback import FallbackHandler
from fallback.web_search import WebSearcher
from fallback.supplementary import SupplementaryRetriever


@pytest.fixture
def handler():
    """创建测试用 FallbackHandler（注入真实 WebSearcher + SupplementaryRetriever）"""
    return FallbackHandler(
        web_searcher=WebSearcher(),
        supplementary=SupplementaryRetriever(),
    )


class TestFallbackHandler:
    @pytest.mark.asyncio
    async def test_handle_web_search_disabled_returns_no_answer(self, monkeypatch, handler):
        """联网搜索禁用时直接返回诚实告知"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", False)
        ctx = PipelineContext(query="测试问题")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.fallback_level == FallbackLevel.NO_ANSWER
        assert result.is_fallback is True
        assert result.confidence == 0.0
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_handle_uses_no_answer_message(self, monkeypatch, handler):
        """验证使用配置中的 no_answer_message"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", False)
        monkeypatch.setattr(
            "fallback.handler.settings.fallback.no_answer_message",
            "自定义兜底消息"
        )
        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.answer == "自定义兜底消息"

    @pytest.mark.asyncio
    async def test_handle_web_search_enabled_placeholder_returns_no_answer(self, monkeypatch, handler):
        """联网搜索启用但返回 None 时，回退到诚实告知"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", True)

        async def mock_search(query):
            return None

        handler._web_searcher.search = mock_search

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.fallback_level == FallbackLevel.NO_ANSWER
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_handle_web_search_success(self, monkeypatch, handler):
        """联网搜索成功时返回搜索结果"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", True)

        async def mock_search(query):
            return "从网络搜索获取的答案"

        handler._web_searcher.search = mock_search

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.fallback_level == FallbackLevel.WEB_SEARCH
        assert result.is_fallback is True
        assert result.answer == "从网络搜索获取的答案"

    @pytest.mark.asyncio
    async def test_handle_web_search_exception_falls_back(self, monkeypatch, handler):
        """联网搜索异常时回退到诚实告知"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", True)

        async def failing_search(query):
            raise RuntimeError("搜索服务不可用")

        handler._web_searcher.search = failing_search

        ctx = PipelineContext(query="测试")
        ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await handler.handle(ctx)
        assert result.fallback_level == FallbackLevel.NO_ANSWER
