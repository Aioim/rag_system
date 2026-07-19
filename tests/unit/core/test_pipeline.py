"""RAGPipeline 测试"""
import tempfile
from pathlib import Path

import pytest
from models.context import PipelineContext
from models.enums import FallbackLevel, Intent, RetrievalEval
from session.store import SessionStore
from session.manager import SessionManager


class FakeLLM:
    """模拟 LLM — 提供 generate 和 ainvoke 方法"""

    def __init__(self, response="测试回答"):
        self.response = response
        self.calls = []
        # 可配置的意图分类响应（默认清晰）
        self.intent_response = (
            '{"intent": "concept", "is_clear": true, "clarification_question": null}'
        )

    async def generate(self, prompt, **kwargs):
        self.calls.append(("generate", prompt, kwargs))
        # 根据 prompt 内容返回不同响应（匹配 IntentClassifier 的 prompt 标识）
        if "查询意图分类器" in prompt:
            return self.intent_response
        if "对话上下文理解" in prompt:
            return self.response
        if "假设性答案" in prompt:
            return "假设答案文本"
        if "关键词" in prompt:
            return "关键词1 关键词2"
        if "同义" in prompt:
            return "同义变体A\n同义变体B"
        return self.response

    async def ainvoke(self, prompt, **kwargs):
        self.calls.append(("ainvoke", prompt, kwargs))

        class FakeMsg:
            content = "测试回答内容"

        return FakeMsg()


@pytest.fixture
def session_manager():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()


class TestRAGPipelineRun:
    @pytest.mark.asyncio
    async def test_run_normal_flow(self, session_manager, monkeypatch):
        """正常流程：查询理解 → 检索 → 生成 → 返回结果"""
        # Mock 检索层
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )
        # Mock 生成层
        monkeypatch.setattr(
            "core.pipeline.GenerationLayer",
            lambda llm: _FakeGenerationLayer(answer="RAG是检索增强生成技术"),
        )

        from core.pipeline import RAGPipeline

        llm = FakeLLM()
        pipeline = RAGPipeline(llm, session_manager)
        ctx = await pipeline.run("什么是RAG？")

        assert ctx.query == "什么是RAG？"
        assert ctx.answer == "RAG是检索增强生成技术"
        assert ctx.retrieval_eval == RetrievalEval.SUFFICIENT
        assert ctx.needs_clarification is False
        assert "pipeline_ms" in ctx.metadata

    @pytest.mark.asyncio
    async def test_run_needs_clarification_short_circuits(self, session_manager):
        """模糊问题短路返回，不调用检索和生成"""
        from core.pipeline import RAGPipeline

        llm = FakeLLM()
        pipeline = RAGPipeline(llm, session_manager)

        # 直接 mock query_layer.process 返回需要澄清的 ctx
        async def mock_process(query, session_id=None, collection="default"):
            ctx = PipelineContext(query=query, collection=collection)
            ctx.needs_clarification = True
            ctx.clarification_question = "您想了解哪方面内容？"
            ctx.intent = Intent.CONCEPT
            return ctx

        pipeline.query_layer.process = mock_process

        ctx = await pipeline.run("帮帮我")

        assert ctx.needs_clarification is True
        assert ctx.clarification_question == "您想了解哪方面内容？"
        # 短路后不应有答案
        assert ctx.answer == ""
        assert "pipeline_ms" in ctx.metadata

    @pytest.mark.asyncio
    async def test_run_insufficient_triggers_fallback(self, session_manager, monkeypatch):
        """检索不足时触发兜底处理（联网搜索失败 → 诚实告知）"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=False),
        )
        monkeypatch.setattr(
            "fallback.web_search.settings.web_search.enabled", False
        )

        from core.pipeline import RAGPipeline

        llm = FakeLLM()
        pipeline = RAGPipeline(llm, session_manager)
        ctx = await pipeline.run("非常冷门的问题")

        assert ctx.retrieval_eval == RetrievalEval.INSUFFICIENT
        assert ctx.is_fallback is True
        assert ctx.fallback_level == FallbackLevel.NO_ANSWER
        assert len(ctx.answer) > 0

    @pytest.mark.asyncio
    async def test_run_with_session_saves_messages(self, session_manager, monkeypatch):
        """多轮对话时保存用户问题和助手回答到会话"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )
        monkeypatch.setattr(
            "core.pipeline.GenerationLayer",
            lambda llm: _FakeGenerationLayer(answer="测试回答"),
        )

        from core.pipeline import RAGPipeline

        session_manager.get_or_create("s1")
        llm = FakeLLM()
        pipeline = RAGPipeline(llm, session_manager)
        await pipeline.run("测试问题", session_id="s1")

        session = session_manager.get("s1")
        messages = session.messages
        assert len(messages) >= 2
        assert messages[-2].role == "user"
        assert messages[-2].content == "测试问题"
        assert messages[-1].role == "assistant"
        assert messages[-1].content == "测试回答"

    @pytest.mark.asyncio
    async def test_run_query_layer_exception_graceful_degradation(self, session_manager, monkeypatch):
        """查询理解层异常时降级继续"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )
        monkeypatch.setattr(
            "core.pipeline.GenerationLayer",
            lambda llm: _FakeGenerationLayer(answer="降级回答"),
        )

        from core.pipeline import RAGPipeline

        # 创建一个会使 query_layer.process 失败的场景
        llm = FakeLLM()

        async def failing_generate(prompt, **kwargs):
            raise RuntimeError("LLM不可用")

        llm.generate = failing_generate

        pipeline = RAGPipeline(llm, session_manager)
        ctx = await pipeline.run("测试问题")

        # 降级后仍应有答案
        assert ctx.answer == "降级回答"
        assert ctx.needs_clarification is False

    @pytest.mark.asyncio
    async def test_run_generation_exception_returns_fallback_message(self, session_manager, monkeypatch):
        """生成层异常时返回兜底消息"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )

        from core.pipeline import RAGPipeline

        llm = FakeLLM()
        pipeline = RAGPipeline(llm, session_manager)

        # 让生成层抛异常
        async def failing_generate(ctx):
            raise RuntimeError("LLM生成失败")

        pipeline.generation_layer.generate = failing_generate

        ctx = await pipeline.run("测试问题")

        assert ctx.is_fallback is True
        assert ctx.fallback_level == FallbackLevel.NO_ANSWER
        assert ctx.confidence == 0.0
        assert len(ctx.answer) > 0

    @pytest.mark.asyncio
    async def test_run_with_collection(self, session_manager, monkeypatch):
        """collection 参数正确传递"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )
        monkeypatch.setattr(
            "core.pipeline.GenerationLayer",
            lambda llm: _FakeGenerationLayer(),
        )

        from core.pipeline import RAGPipeline

        llm = FakeLLM()
        pipeline = RAGPipeline(llm, session_manager)
        ctx = await pipeline.run("查询", collection="tech_docs")
        assert ctx.collection == "tech_docs"

    @pytest.mark.asyncio
    async def test_run_need_more_triggers_supplementary_retrieval(self, session_manager, monkeypatch):
        """NEED_MORE 时触发补充检索并继续生成"""
        # 第一次检索返回 NEED_MORE，第二次（补充检索）返回 SUFFICIENT
        call_count = [0]

        class _TwoPhaseRetrievalLayer:
            async def retrieve(self, ctx, top_k=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    ctx.retrieval_eval = RetrievalEval.NEED_MORE
                else:
                    ctx.retrieval_eval = RetrievalEval.SUFFICIENT
                return ctx

        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _TwoPhaseRetrievalLayer(),
        )
        monkeypatch.setattr(
            "core.pipeline.GenerationLayer",
            lambda llm: _FakeGenerationLayer(answer="补充检索后的回答"),
        )

        from core.pipeline import RAGPipeline

        llm = FakeLLM()
        pipeline = RAGPipeline(llm, session_manager)
        ctx = await pipeline.run("需要更多资料的问题")

        # NEED_MORE 触发补充检索，标记 PARTIAL，继续生成
        assert ctx.answer == "补充检索后的回答"

    @pytest.mark.asyncio
    async def test_run_metadata_tracks_pipeline_time(self, session_manager, monkeypatch):
        """验证 metadata 中包含 pipeline_ms"""
        monkeypatch.setattr(
            "core.pipeline.get_retrieval_layer",
            lambda: _FakeRetrievalLayer(sufficient=True),
        )
        monkeypatch.setattr(
            "core.pipeline.GenerationLayer",
            lambda llm: _FakeGenerationLayer(),
        )

        from core.pipeline import RAGPipeline

        llm = FakeLLM()
        pipeline = RAGPipeline(llm, session_manager)
        ctx = await pipeline.run("测试")

        assert "pipeline_ms" in ctx.metadata
        assert ctx.metadata["pipeline_ms"] >= 0


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


class TestSaveToSessionOffEventLoop:
    """审查 H7：会话写入不应在事件循环线程中执行"""

    @pytest.mark.asyncio
    async def test_add_message_runs_off_event_loop(self):
        # Arrange
        import threading
        from types import SimpleNamespace
        from core.pipeline import RAGPipeline

        loop_thread = threading.current_thread()
        seen_threads: list = []

        class RecordingSM:
            def add_message(self, sid, role, content):
                seen_threads.append(threading.current_thread())

        fake_self = SimpleNamespace(_session_manager=RecordingSM())

        # Act — _save_to_session 应为可等待的异步方法
        await RAGPipeline._save_to_session(fake_self, "s1", "问", "答")

        # Assert
        assert len(seen_threads) == 2, "user/assistant 各写一条"
        assert all(t is not loop_thread for t in seen_threads), (
            "SQLite 会话写入应通过 to_thread 移出事件循环线程"
        )
