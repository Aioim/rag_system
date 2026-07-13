"""KeywordRewriter 测试"""
import pytest
from query.rewriters.keyword import KeywordRewriter


class MockLLM:
    def __init__(self, response="关键词1 关键词2"):
        self.response = response
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


class TestKeywordRewriter:
    @pytest.mark.asyncio
    async def test_rewrite_returns_keywords(self):
        llm = MockLLM(response="年假 申请 材料 流程")
        rewriter = KeywordRewriter(llm)
        result = await rewriter.rewrite("如何申请年假？")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "年假" in result[0]

    @pytest.mark.asyncio
    async def test_rewrite_handles_empty_response(self):
        llm = MockLLM(response="")
        rewriter = KeywordRewriter(llm)
        result = await rewriter.rewrite("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_rewrite_has_query_in_prompt(self):
        llm = MockLLM()
        rewriter = KeywordRewriter(llm)
        await rewriter.rewrite("五险一金缴纳比例")
        prompt = llm.calls[0][0]
        assert "五险一金缴纳比例" in prompt
