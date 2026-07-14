"""HyDERewriter 测试"""
import pytest
from query.rewriters.hyde import HyDERewriter
from tests.unit.query.conftest import MockLLM


class TestHyDERewriter:
    @pytest.mark.asyncio
    async def test_rewrite_returns_single_element_list(self):
        llm = MockLLM(response="RAG是一种结合检索和生成的AI技术，可以有效提供基于知识库的问答服务。")
        rewriter = HyDERewriter(llm)
        result = await rewriter.rewrite("什么是RAG？")
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) > 0

    @pytest.mark.asyncio
    async def test_rewrite_contains_query_context(self):
        """假设答案应与原始问题相关（基础检查）"""
        llm = MockLLM()
        rewriter = HyDERewriter(llm)
        result = await rewriter.rewrite("如何申请年假？")
        prompt = llm.calls[0][0]
        assert "申请年假" in prompt

    @pytest.mark.asyncio
    async def test_rewrite_on_llm_error(self):

        class FailingLLM:
            async def ainvoke(self, prompt, **kwargs):
                raise RuntimeError("timeout")

        rewriter = HyDERewriter(FailingLLM())
        result = await rewriter.rewrite("test")
        assert result == []
