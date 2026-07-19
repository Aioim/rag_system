"""PipelineContext 测试"""
from models.context import PipelineContext


class TestPipelineContext:
    def test_minimal(self):
        ctx = PipelineContext(query="测试问题")
        assert ctx.query == "测试问题"
        assert ctx.rewritten_queries == []
        assert ctx.intent is None
        assert ctx.candidates == []
        assert ctx.answer == ""
        assert ctx.confidence == 0.0
        assert ctx.is_fallback is False

    def test_with_results(self):
        ctx = PipelineContext(
            query="问题",
            answer="答案",
            confidence=0.85,
            retrieval_eval="sufficient",
        )
        assert ctx.answer == "答案"
        assert ctx.confidence == 0.85
        assert ctx.retrieval_eval == "sufficient"
