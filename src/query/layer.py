"""查询理解层主编排器 — Pipeline 链式编排"""
import asyncio

from logger import logger
from models.context import PipelineContext
from models.llm import LLMProtocol
from query.aliases import resolve_aliases_in_text
from query.context_fuser import ContextFuser
from query.intent_classifier import IntentClassifier
from query.rewriters import QueryRewriter
from session.manager import SessionManager


class QueryUnderstandingLayer:
    """查询理解层 — 别名映射 → 意图分类 → 上下文融合 → 查询改写"""

    def __init__(self, llm: LLMProtocol, session_manager: SessionManager | None = None) -> None:
        self.intent_classifier = IntentClassifier(llm, temperature=0)
        self.context_fuser = ContextFuser(llm, temperature=0)
        self.rewriter = QueryRewriter(llm)
        self._session_manager = session_manager

    async def process(
        self,
        query: str,
        session_id: str | None = None,
        collection: str = "default",
    ) -> PipelineContext:
        ctx = PipelineContext(query=query, collection=collection)
        ctx.original_query = query

        # 1. 别名映射（解析失败衰减回原始 query，不阻塞 Pipeline）
        try:
            query = resolve_aliases_in_text(query)
        except Exception:
            logger.warning("别名映射解析失败，使用原始 query 降级")
        ctx.query = query

        # 2. 意图分类 + 清晰度判断
        result = await self.intent_classifier.classify(query)
        ctx.intent = result.intent
        if not result.is_clear:
            ctx.needs_clarification = True
            ctx.clarification_question = result.clarification_question
            # 短路时也尝试获取 session（用于后续记录澄清交互）
            if session_id and self._session_manager is not None:
                # 同步 SQLite 读取移出事件循环线程
                ctx.session = await asyncio.to_thread(
                    self._session_manager.get, session_id
                )
            return ctx

        # 3. 多轮上下文融合
        if session_id and self._session_manager is not None:
            # 同步 SQLite 读取移出事件循环线程
            session = await asyncio.to_thread(
                self._session_manager.get, session_id
            )
            ctx.session = session
            if session is not None:
                fused_query = await self.context_fuser.fuse(query, session)
                ctx.query = fused_query
                query = fused_query

        # 4. 查询改写（并行）
        ctx.rewritten_queries = await self.rewriter.rewrite(query)
        return ctx


# ============================================================================
# 自测：用 Mock LLM 演示完整 Pipeline
# ============================================================================
if __name__ == "__main__":
    import asyncio
    from types import SimpleNamespace

    class _MockLLM:
        """可编程 Mock LLM — 根据 prompt 内容返回不同响应"""

        async def ainvoke(self, prompt, **_kw):
            if "查询意图分类器" in prompt:
                return SimpleNamespace(content='{"intent": "concept", "is_clear": true, "clarification_question": null}')
            if "对话上下文理解" in prompt:
                return SimpleNamespace(content="RAG架构中检索和生成的协作方式")
            if "假设性答案" in prompt:
                return SimpleNamespace(content="RAG（检索增强生成）是一种结合信息检索和文本生成的AI架构...")
            if "关键词" in prompt:
                return SimpleNamespace(content="RAG 检索 增强 生成 架构")
            if "同义" in prompt:
                return SimpleNamespace(content="检索增强生成的原理\nRAG技术的工作机制")
            return SimpleNamespace(content="default")

    class _MockSession:
        def __init__(self, sid):
            self.session_id = sid


    async def main():
        # 最简单的 Mock SessionManager
        sm = SimpleNamespace(get=lambda sid: _MockSession(sid))

        llm = _MockLLM()
        layer = QueryUnderstandingLayer(llm, sm)

        print("=" * 60)
        print("QueryUnderstandingLayer 自测")
        print("=" * 60)

        ctx = await layer.process("什么是RAG架构？")
        print(f"  original_query : {ctx.original_query}")
        print(f"  query          : {ctx.query}")
        print(f"  intent         : {ctx.intent.value if ctx.intent else 'N/A'}")
        print(f"  rewritten      : {ctx.rewritten_queries}")
        print(f"  needs_clarify  : {ctx.needs_clarification}")

    asyncio.run(main())
