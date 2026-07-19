"""WebSearcher 测试"""
import pytest
from fallback.web_search import WebSearcher


class TestWebSearcher:
    @pytest.mark.asyncio
    async def test_search_disabled_returns_none(self, monkeypatch):
        """联网搜索禁用时返回 None"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", False)
        searcher = WebSearcher()
        result = await searcher.search("测试查询")
        assert result is None

    @pytest.mark.asyncio
    async def test_search_empty_results_returns_none(self, monkeypatch):
        """搜索无结果时返回 None"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", True)

        from fallback import web_search as ws_mod

        async def mock_do_search(self, query):
            return []

        monkeypatch.setattr(ws_mod.WebSearcher, "_do_search", mock_do_search)

        searcher = WebSearcher()
        result = await searcher.search("测试")
        assert result is None

    @pytest.mark.asyncio
    async def test_search_success_returns_formatted_text(self, monkeypatch):
        """搜索成功返回格式化文本"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", True)

        from fallback import web_search as ws_mod

        async def mock_do_search(self, query):
            return [
                {"title": "标题1", "body": "内容1", "href": "http://a.com"},
                {"title": "标题2", "body": "内容2", "href": "http://b.com"},
            ]

        monkeypatch.setattr(ws_mod.WebSearcher, "_do_search", mock_do_search)

        searcher = WebSearcher()
        result = await searcher.search("测试")
        assert result is not None
        assert "标题1" in result
        assert "内容1" in result
        assert "http://a.com" in result
        assert "标题2" in result

    @pytest.mark.asyncio
    async def test_search_exception_returns_none(self, monkeypatch):
        """搜索异常时返回 None"""
        monkeypatch.setattr("fallback.web_search.settings.web_search.enabled", True)

        from fallback import web_search as ws_mod

        async def failing_search(self, query):
            raise RuntimeError("搜索服务不可用")

        monkeypatch.setattr(ws_mod.WebSearcher, "_do_search", failing_search)

        searcher = WebSearcher()
        result = await searcher.search("测试")
        assert result is None

    def test_format_results(self):
        """验证搜索结果格式化"""
        results = [
            {"title": "标题A", "body": "正文A", "href": "http://a.com"},
        ]
        text = WebSearcher._format_results(results)
        assert "[1] 标题A" in text
        assert "正文A" in text
        assert "来源: http://a.com" in text
