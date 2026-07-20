"""RAGPipeline ReAct 模式集成测试"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.pipeline import RAGPipeline
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval


class ProgrammableLLM:
    """每次 ainvoke 按顺序返回 preset 响应"""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def ainvoke(self, prompt: str, temperature: float = 0.0):
        if self.call_count >= len(self.responses):
            return MagicMock(content="THOUGHT: done\nACTION: FINISH")
        resp = self.responses[self.call_count]
        self.call_count += 1
        return MagicMock(content=resp)


class TestPipelineReactMode:
    @pytest.fixture(autouse=True)
    def cleanup_singletons(self):
        """重置所有全局单例，保证测试隔离"""
        from agent import reset_react_agent
        from core import reset_rag_pipeline
        reset_rag_pipeline()
        reset_react_agent()
        yield
        reset_rag_pipeline()
        reset_react_agent()
        from agent import reset_react_agent
        from core import reset_rag_pipeline
        from retrieval import reset_retrieval_layer

        reset_react_agent()
        reset_rag_pipeline()
        reset_retrieval_layer()

    @pytest.fixture
    def mock_session_manager(self):
        sm = MagicMock()
        sm.add_message = MagicMock()
        sm.get = MagicMock(return_value=None)
        return sm

    @pytest.mark.asyncio
    async def test_react_mode_single_round(self, mock_session_manager, monkeypatch):
        """ReAct 模式单轮 FINISH 后生成答案"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )
        monkeypatch.setattr(
            "fallback.web_search.settings.web_search.enabled", False
        )

        llm = ProgrammableLLM([
            # Agent 决策：直接回答
            "THOUGHT: 简单问候，无需检索\nACTION: FINISH",
            # 生成答案
            "你好！有什么可以帮助你的？",
        ])
        pipeline = RAGPipeline(llm, mock_session_manager)

        ctx = await pipeline.run("你好", mode="react")

        assert ctx.mode == "react"
        assert len(ctx.react_traces) == 1
        assert ctx.react_traces[0].action == "finish"
        assert ctx.answer == "你好！有什么可以帮助你的？"

    @pytest.mark.asyncio
    async def test_react_mode_falls_back_to_linear_on_error(self, mock_session_manager, monkeypatch):
        """Agent 构造/异常时降级到 linear 模式"""
        from core.pipeline import RAGPipeline

        # 先创建 pipeline（此时 get_retrieval_layer 正常）
        # 然后将 retrieval.get_retrieval_layer 设为不可用，使 get_react_agent 构造失败
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))

        pipeline = RAGPipeline(llm, mock_session_manager)

        monkeypatch.setattr(
            "retrieval.get_retrieval_layer",
            lambda: (_ for _ in ()).throw(RuntimeError("Retrieval layer unavailable")),
        )

        ctx = await pipeline.run("test", mode="react")

        # 降级后应走 linear 流程
        assert ctx.mode == "linear"
        # 降级后的 linear 流程可能触发 web_search（取决于环境），或完全失败
        assert ctx.fallback_level in (FallbackLevel.NO_ANSWER, FallbackLevel.NONE, FallbackLevel.WEB_SEARCH)

    @pytest.mark.asyncio
    async def test_linear_mode_unchanged(self, mock_session_manager, monkeypatch):
        """mode='linear' 时行为不变"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )
        monkeypatch.setattr(
            "core.pipeline.GenerationLayer",
            lambda llm: _FakeGenerationLayer(answer="answer"),
        )

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))

        pipeline = RAGPipeline(llm, mock_session_manager)
        ctx = await pipeline.run("test", mode="linear")

        assert ctx.mode == "linear"
        assert ctx.react_traces == []

    @pytest.mark.asyncio
    async def test_react_mode_agent_search_then_generate(self, mock_session_manager, monkeypatch):
        """ReAct 模式：search → 合并检索 → 生成"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )
        monkeypatch.setattr(
            "fallback.web_search.settings.web_search.enabled", False
        )

        llm = ProgrammableLLM([
            # 第1步：搜索
            "THOUGHT: 需要查询知识库\nACTION: search\nQUERY: RAG定义",
            # 第2步：生成答案（直接 FINISH）
            "THOUGHT: 已获取足够信息\nACTION: FINISH",
            # 生成层调用
            "RAG是检索增强生成技术",
        ])
        pipeline = RAGPipeline(llm, mock_session_manager)

        ctx = await pipeline.run("什么是RAG？", mode="react")

        assert ctx.mode == "react"
        assert len(ctx.react_traces) == 2
        assert ctx.react_traces[0].action == "search"
        assert ctx.react_traces[0].query == "RAG定义"
        assert ctx.react_traces[1].action == "finish"
        assert ctx.answer == "RAG是检索增强生成技术"
        assert "RAG定义" in ctx.rewritten_queries


# ---- Fake Implementations ------------------------------------------------


class _FakeRetrievalLayer:
    """模拟检索层"""

    def __init__(self, sufficient=True, eval_result=None):
        self._sufficient = sufficient
        self._eval_result = eval_result

    async def retrieve(self, ctx, top_k=None):
        if self._eval_result is not None:
            ctx.retrieval_eval = self._eval_result
        else:
            ctx.retrieval_eval = (
                RetrievalEval.SUFFICIENT if self._sufficient else RetrievalEval.INSUFFICIENT
            )
        return ctx


class _FakeGenerationLayer:
    """模拟生成层"""

    def __init__(self, answer="默认回答"):
        self._answer = answer

    async def generate(self, ctx):
        ctx.answer = self._answer
        ctx.confidence = 0.9
        ctx.sources = []
        return ctx
