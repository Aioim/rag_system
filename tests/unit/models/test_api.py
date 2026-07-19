"""API models 测试"""
from models.api import (
    ChatRequest, ChatResponse,
    SearchRequest, SearchResponse, Source,
)


class TestSource:
    def test_construction(self):
        s = Source(doc_id="d1", doc_title="报告", chunk_text="内容...", score=0.95)
        assert s.score == 0.95
        assert s.doc_title == "报告"


class TestChatRequest:
    def test_defaults(self):
        req = ChatRequest(query="什么是RAG？")
        assert req.session_id is None
        assert req.stream is False
        assert req.top_k == 5

    def test_with_session(self):
        req = ChatRequest(query="继续", session_id="s1", top_k=3)
        assert req.session_id == "s1"
        assert req.top_k == 3


class TestChatResponse:
    def test_defaults(self):
        resp = ChatResponse(answer="RAG是...")
        assert resp.sources == []
        assert resp.confidence == 0.0
        assert resp.is_fallback is False


class TestSearchRequest:
    def test_defaults(self):
        req = SearchRequest(query="关键词")
        assert req.collection == "default"
        assert req.top_k == 10


class TestSearchResponse:
    def test_defaults(self):
        resp = SearchResponse()
        assert resp.results == []
        assert resp.search_type == "hybrid"
