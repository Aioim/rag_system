"""查询理解层主编排器 — Pipeline 链式编排"""
from config.aliases import resolve_aliases_in_text
from models.context import PipelineContext
from session.manager import SessionManager
from query.intent_classifier import IntentClassifier
from query.context_fuser import ContextFuser
from query.rewriters import QueryRewriter


class QueryUnderstandingLayer:
    """查询理解层 — 别名映射 → 意图分类 → 上下文融合 → 查询改写"""

    def __init__(self, llm, session_manager: SessionManager):
        self.intent_classifier = IntentClassifier(llm)
        self.context_fuser = ContextFuser(llm, session_manager)
        self.rewriter = QueryRewriter(llm)
        self._session_manager = session_manager

    async def process(
        self,
        query: str,
        session_id: str | None = None,
        collection: str = "default",
    ) -> PipelineContext:
        ctx = PipelineContext(query=query, collection=collection)

        # 1. 别名映射
        query = resolve_aliases_in_text(query)

        # 2. 意图分类 + 清晰度判断
        result = await self.intent_classifier.classify(query)
        ctx.intent = result.intent
        if not result.is_clear:
            ctx.needs_clarification = True
            ctx.clarification_question = result.clarification_question
            return ctx  # 短路返回

        # 3. 多轮上下文融合
        if session_id:
            query = await self.context_fuser.fuse(query, session_id)
            ctx.query = query
            ctx.session = self._session_manager.get(session_id)

        # 4. 查询改写（并行）
        ctx.rewritten_queries = await self.rewriter.rewrite(query)
        return ctx
