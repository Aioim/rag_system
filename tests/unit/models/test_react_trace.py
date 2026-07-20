"""ReActTrace 测试"""
from models.react_trace import ReActTrace


class TestReActTrace:
    def test_minimal(self):
        t = ReActTrace(iteration=1, thought="test", action="search")
        assert t.iteration == 1
        assert t.thought == "test"
        assert t.action == "search"
        assert t.query is None
        assert t.observation is None
        assert t.elapsed_ms == 0.0

    def test_full(self):
        t = ReActTrace(
            iteration=2,
            thought="我需要搜索",
            action="web_search",
            query="RAG是什么",
            observation="检索结果",
            elapsed_ms=150.5,
        )
        assert t.iteration == 2
        assert t.thought == "我需要搜索"
        assert t.action == "web_search"
        assert t.query == "RAG是什么"
        assert t.observation == "检索结果"
        assert t.elapsed_ms == 150.5
