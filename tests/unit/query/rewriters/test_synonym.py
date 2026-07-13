"""SynonymRewriter 测试"""
import pytest
from query.rewriters.synonym import SynonymRewriter


class MockLLM:
    def __init__(self, response="同义表达1\n同义表达2"):
        self.response = response
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


class TestSynonymRewriter:
    @pytest.mark.asyncio
    async def test_rewrite_returns_variants(self):
        llm = MockLLM(response="怎样申请年假\n如何办理带薪年休假")
        rewriter = SynonymRewriter(llm)
        result = await rewriter.rewrite("如何申请年假？")
        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_rewrite_splits_multiline_response(self):
        llm = MockLLM(response="变体A\n变体B\n变体C")
        rewriter = SynonymRewriter(llm)
        result = await rewriter.rewrite("原始查询")
        assert len(result) == 3
        assert "变体A" in result
        assert "变体B" in result
        assert "变体C" in result

    @pytest.mark.asyncio
    async def test_rewrite_filters_empty_lines(self):
        llm = MockLLM(response="变体A\n\n\n变体B\n  \n")
        rewriter = SynonymRewriter(llm)
        result = await rewriter.rewrite("test")
        assert len(result) == 2
