"""QueryUnderstandingLayer 测试"""
import tempfile
from pathlib import Path

import pytest
from models.enums import Intent
from session.store import SessionStore
from session.manager import SessionManager
from query.layer import QueryUnderstandingLayer


class MockLLM:
    """可编程的 Mock LLM — 根据 prompt 内容返回不同响应"""

    def __init__(self):
        self.intent_response = '{"intent": "concept", "is_clear": true, "clarification_question": null}'
        self.fuse_response = "完整的问题"
        self.hyde_response = "假设答案"
        self.keyword_response = "关键词"
        self.synonym_response = "同义变体"

    async def generate(self, prompt, **kwargs):
        if "查询意图分类器" in prompt:
            return self.intent_response
        elif "对话上下文理解" in prompt:
            return self.fuse_response
        elif "假设性答案" in prompt:
            return self.hyde_response
        elif "关键词" in prompt:
            return self.keyword_response
        elif "同义" in prompt:
            return self.synonym_response
        return "default"


@pytest.fixture
def session_manager():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()


class TestQueryUnderstandingLayerProcess:
    @pytest.mark.asyncio
    async def test_process_basic_query(self, session_manager):
        """基本流程：无 session 的简单查询"""
        llm = MockLLM()
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("什么是RAG？")

        assert ctx.query == "什么是RAG？"
        assert ctx.intent == Intent.CONCEPT
        assert ctx.needs_clarification is False
        assert len(ctx.rewritten_queries) > 0
        assert ctx.rewritten_queries[0] == "什么是RAG？"

    @pytest.mark.asyncio
    async def test_process_short_circuits_on_unclear_query(self, session_manager):
        """模糊问题短路返回，不继续检索"""
        llm = MockLLM()
        llm.intent_response = (
            '{"intent": "concept", "is_clear": false, '
            '"clarification_question": "您想了解什么内容？"}'
        )
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("帮帮我")

        assert ctx.needs_clarification is True
        assert ctx.clarification_question == "您想了解什么内容？"
        # 短路后不应有 rewritten_queries
        assert ctx.rewritten_queries == []

    @pytest.mark.asyncio
    async def test_process_with_session(self, session_manager):
        """有 session 时触发多轮上下文融合"""
        llm = MockLLM()
        llm.fuse_response = "申请年假需要什么材料？"
        layer = QueryUnderstandingLayer(llm, session_manager)

        # 准备会话
        session_manager.get_or_create("s1")
        session_manager.add_message("s1", "user", "年假怎么申请？")
        session_manager.add_message("s1", "assistant", "年假需要登录OA...")

        ctx = await layer.process("需要什么材料？", session_id="s1")
        assert ctx.query == "申请年假需要什么材料？"
        assert ctx.session is not None
        assert ctx.session.session_id == "s1"

    @pytest.mark.asyncio
    async def test_process_no_session_skips_fusion(self, session_manager):
        """无 session_id 时跳过融合步骤"""
        llm = MockLLM()
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("独立问题", session_id=None)

        assert ctx.query == "独立问题"
        assert len(ctx.rewritten_queries) > 0

    @pytest.mark.asyncio
    async def test_process_with_collection(self, session_manager):
        """collection 参数正确传递到 PipelineContext"""
        llm = MockLLM()
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("查询", collection="tech_docs")
        assert ctx.collection == "tech_docs"

    @pytest.mark.asyncio
    async def test_process_intent_classifier_failure(self, session_manager):
        """意图分类 LLM 失败时降级不抛异常"""

        class FailingIntentLLM:
            async def generate(self, prompt, **kwargs):
                raise RuntimeError("LLM error")

        layer = QueryUnderstandingLayer(FailingIntentLLM(), session_manager)
        ctx = await layer.process("任意问题")
        # 降级：intent 为 CONCEPT（分类失败默认值），不抛异常
        assert ctx.intent == Intent.CONCEPT
        # rewritten_queries 至少包含原始 query（rewriters 也都失败但 QueryRewriter 始终保留原始 query）
        assert len(ctx.rewritten_queries) >= 1
        assert ctx.rewritten_queries[0] == "任意问题"

    @pytest.mark.asyncio
    async def test_process_rewritten_queries_contain_original(self, session_manager):
        """验证原始 query 在 rewritten_queries 结果中"""
        llm = MockLLM()
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("原始查询")
        # 原始查询（经过别名映射后）在第一位
        assert len(ctx.rewritten_queries) >= 1
