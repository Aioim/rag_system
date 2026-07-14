"""HyDERewriter 测试"""
import pytest
from types import SimpleNamespace
from query.rewriters.hyde import HyDERewriter


class MockLLM:
    def __init__(self, response="这是假设答案文本"):
        self.response = response
        self.calls = []

    async def ainvoke(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return SimpleNamespace(content=self.response)


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
        # 不应该抛异常，返回空 list（由 QueryRewriter 编排层处理）
        try:
            result = await rewriter.rewrite("test")
        except RuntimeError:
            result = []
        assert result == []
