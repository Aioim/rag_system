"""RAGPipeline — 核心 Pipeline 主编排器

串联完整 RAG 问答链路：
  查询理解 → 检索 → (检索不足→兜底) → 生成 → 会话记录

各层异常独立降级，不中断 Pipeline。
"""
import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fallback.handler import FallbackHandler
    from models.llm import LLMProtocol

from fallback import get_fallback_handler
from generation.layer import GenerationLayer
from logger import logger
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval
from query.layer import QueryUnderstandingLayer
from retrieval import get_retrieval_layer
from session.manager import SessionManager


class RAGPipeline:
    """RAG 核心 Pipeline

    编排完整的 RAG 问答链路：
    1. 查询理解 — 意图分类、指代消解、查询改写
    2. 混合检索 — 向量+BM25 → RRF → Rerank → Self-RAG 自评
    3. 兜底处理 — NEED_MORE→补充检索；INSUFFICIENT→联网搜索/诚实告知
    4. 生成回答 — Prompt 组装 → LLM 生成 → 事实核查 → 引用标注
    5. 会话记录 — 写入本轮对话历史

    每层异常时降级继续，不阻塞 Pipeline。
    """

    def __init__(self, llm: "LLMProtocol", session_manager: SessionManager) -> None:
        self._llm = llm
        self._session_manager = session_manager
        self.query_layer = QueryUnderstandingLayer(llm, session_manager)
        self.retrieval_layer = get_retrieval_layer()
        self.generation_layer = GenerationLayer(llm)
        self.fallback: "FallbackHandler" = get_fallback_handler()

    async def run(
        self,
        query: str,
        session_id: str | None = None,
        collection: str = "default",
        mode: str = "linear",
        max_iterations: int = 5,
        show_reasoning: bool = False,
    ) -> PipelineContext:
        """执行完整 RAG 问答链路

        Args:
            query: 用户问题
            session_id: 会话 ID（多轮对话使用）
            collection: 知识库集合名称
            mode: 运行模式 ("linear" | "react")
            max_iterations: ReAct 最大迭代次数
            show_reasoning: 是否在 metadata 中保留推理过程

        Returns:
            PipelineContext
        """
        t0 = time.perf_counter()
        ctx = PipelineContext(query=query, collection=collection)
        ctx.mode = mode
        ctx.max_iterations = max_iterations
        ctx.original_query = query  # 确保异常降级时不会保存空字符串

        if mode == "react":
            return await self._run_react(
                query, session_id, collection, max_iterations, show_reasoning
            )

        # ---- 1. 查询理解 ------------------------------------------------
        try:
            ctx = await self.query_layer.process(query, session_id, collection)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("查询理解层异常，使用原始 query 降级继续")
            # 确保降级时 ctx 处于一致状态
            ctx.query = query
            ctx.original_query = query
            ctx.needs_clarification = False

        if ctx.needs_clarification:
            self._record_elapsed(ctx, t0)
            return ctx

        # ---- 2. 检索 ----------------------------------------------------
        try:
            ctx = await self.retrieval_layer.retrieve(ctx)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("检索层异常，标记为 INSUFFICIENT")
            ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        # ---- 3. 兜底处理 ------------------------------------------------
        ctx, need_short_circuit = await self._apply_fallback(ctx, t0, query, session_id)
        if need_short_circuit:
            return ctx

        # ---- 4. 生成 ----------------------------------------------------
        try:
            ctx = await self.generation_layer.generate(ctx)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("生成层异常，返回兜底消息")
            ctx = self.fallback.no_answer(ctx)

        # ---- 5. 记录 ----------------------------------------------------
        self._record_elapsed(ctx, t0)
        await self._save_to_session(session_id, ctx.original_query, ctx.answer)
        return ctx

    async def _apply_fallback(
        self,
        ctx: PipelineContext,
        t0: float,
        query: str,
        session_id: str | None,
    ) -> tuple[PipelineContext, bool]:
        """兜底处理；返回 (可能更新的 ctx, 短路标志)"""
        if ctx.retrieval_eval is RetrievalEval.NEED_MORE:
            try:
                ctx = await self.fallback.handle(ctx, self.retrieval_layer)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("补充检索异常，标记 PARTIAL 继续生成")
                ctx.is_fallback = True
                ctx.fallback_level = FallbackLevel.PARTIAL

        if ctx.retrieval_eval is RetrievalEval.INSUFFICIENT:
            try:
                ctx = await self.fallback.handle(ctx)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("兜底层异常，降级为诚实告知")
                ctx = self.fallback.no_answer(ctx)

            if ctx.fallback_level in (FallbackLevel.NO_ANSWER, FallbackLevel.WEB_SEARCH):
                self._record_elapsed(ctx, t0)
                await self._save_to_session(session_id, query, ctx.answer)
                return ctx, True

        return ctx, False

    async def _run_react(
        self,
        query: str,
        session_id: str | None,
        collection: str,
        max_iterations: int,
        show_reasoning: bool,
    ) -> PipelineContext:
        """ReAct Agent 分支

        ReAct 模式跳过查询理解层（Agent 自行推理），
        走 Agent 循环 → 合并检索 → 自评 → 兜底 → 生成 → 会话记录。
        """
        from agent import get_react_agent
        from retrieval.evaluator import evaluate as self_evaluate

        t0 = time.perf_counter()
        ctx = PipelineContext(
            query=query, collection=collection,
            mode="react", max_iterations=max_iterations,
        )
        original_query = query  # 保存原始 query，用于会话记录和兜底
        ctx.original_query = original_query

        # ---- 1. 别名映射 ------------------------------------------------
        try:
            from query.aliases import resolve_aliases_in_text
            resolved_query = resolve_aliases_in_text(query)
            ctx.query = resolved_query
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("ReAct 别名映射失败，使用原始 query")
            resolved_query = query

        # ---- 2. ReAct Agent 循环 ----------------------------------------
        try:
            agent = get_react_agent(self._llm, None, None)
            result = await agent.run(resolved_query, collection)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ReAct Agent 异常，降级到 linear 模式")
            return await self.run(
                original_query, session_id, collection, mode="linear"
            )

        ctx.react_traces = result.react_traces
        ctx.reranked = result.reranked

        # ---- 3. 合并检索（Agent 所有 search query 统一用 RetrievalLayer 重新检索）--
        search_queries = [
            t.query for t in result.react_traces
            if t.action == "search" and t.query
        ]
        web_searched = any(
            t.action == "web_search" for t in result.react_traces
        )

        if search_queries and not ctx.reranked:
            try:
                ctx.rewritten_queries = list(dict.fromkeys(search_queries))
                ctx = await self.retrieval_layer.retrieve(ctx)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ReAct 合并检索异常")

        # ---- 4. 检索自评 + 兜底（仅当 Agent 实际执行过检索） --------------
        agent_did_search = bool(search_queries) or web_searched
        if agent_did_search:
            if ctx.retrieval_eval is None:
                ctx.retrieval_eval = self_evaluate(ctx.reranked)

            ctx, need_short_circuit = await self._apply_fallback(ctx, t0, original_query, session_id)
            if need_short_circuit:
                return ctx

        # ---- 5. 生成 ----------------------------------------------------
        try:
            ctx = await self.generation_layer.generate(ctx)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ReAct 生成层异常")
            ctx = self.fallback.no_answer(ctx)

        # ---- 6. 记录 ----------------------------------------------------
        self._record_elapsed(ctx, t0)
        if show_reasoning:
            ctx.metadata["react_traces"] = [
                {
                    "iteration": t.iteration,
                    "thought": t.thought,
                    "action": t.action,
                    "query": t.query,
                    "elapsed_ms": t.elapsed_ms,
                }
                for t in result.react_traces
            ]
        await self._save_to_session(session_id, original_query, ctx.answer)
        return ctx

    async def run_stream(
        self,
        query: str,
        session_id: str | None = None,
        collection: str = "default",
        mode: str = "linear",
        max_iterations: int = 5,
        show_reasoning: bool = False,
    ):
        """流式执行 RAG Pipeline，支持 ReAct 事件推送

        Phase 1: 透传 ReActAgent.run_stream() 事件（含过滤）
        Phase 2: 合并检索 → 生成 → done 事件

        Yields:
            SSEEvent
        """
        from agent.react_agent import SSEEvent

        if mode != "react":
            yield SSEEvent("done", {
                "answer": "streaming only supported for react mode",
                "sources": [], "confidence": 0.0,
            })
            return

        t0 = time.perf_counter()
        search_queries: list[str] = []
        web_searched = False
        original_query = query  # 保存原始 query，用于会话记录

        # 1. 别名映射
        try:
            from query.aliases import resolve_aliases_in_text
            query = resolve_aliases_in_text(query)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("流式别名映射失败")

        # 2. Agent 流式循环：收集事件 + 透传
        try:
            from agent import get_react_agent
            agent = get_react_agent(self._llm, None, None)
            async for event in agent.run_stream(query, collection):
                # 收集 search query 用于后续合并检索
                if event.event == "action" and event.data.get("action") == "search":
                    sq = event.data.get("query", "")
                    if sq:
                        search_queries.append(sq)
                if event.event == "action" and event.data.get("action") == "web_search":
                    web_searched = True
                # 过滤
                if not show_reasoning and event.event in ("thought", "action", "observation"):
                    continue
                yield event
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ReAct Agent 流式异常")
            yield SSEEvent("done", {"answer": "", "sources": [], "confidence": 0.0})
            return

        # 3. 合并检索（用 Agent 搜过的所有 query）
        ctx = PipelineContext(query=query, collection=collection, mode="react")
        ctx.original_query = original_query
        # 注：流式模式下 react_traces 不完整（Agent.run_stream 仅产出事件，
        # 不含完整的 ReActTrace 列表），下游组件不应依赖流式 ctx.react_traces。
        if search_queries:
            try:
                ctx.rewritten_queries = list(dict.fromkeys(search_queries))
                ctx = await self.retrieval_layer.retrieve(ctx)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("流式合并检索异常")

        # 4. 兜底（与 _run_react 保持一致：NEED_MORE → 补充检索；INSUFFICIENT → 联网）
        if ctx.retrieval_eval is RetrievalEval.NEED_MORE:
            try:
                ctx = await self.fallback.handle(ctx, self.retrieval_layer)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("流式补充检索异常")
                ctx.is_fallback = True
                ctx.fallback_level = FallbackLevel.PARTIAL

        if ctx.retrieval_eval is RetrievalEval.INSUFFICIENT and not web_searched:
            try:
                ctx = await self.fallback.handle(ctx)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("流式兜底异常")
                ctx = self.fallback.no_answer(ctx)

        # 兜底短路：联网搜索或诚实告知后跳过生成（与 _run_react 的 _apply_fallback 一致）
        if ctx.fallback_level in (FallbackLevel.NO_ANSWER, FallbackLevel.WEB_SEARCH):
            self._record_elapsed(ctx, t0)
            await self._save_to_session(session_id, original_query, ctx.answer)
            yield SSEEvent("done", {
                "answer": ctx.answer,
                "sources": [
                    {"doc_id": s.doc_id, "doc_title": s.doc_title,
                     "chunk_text": s.chunk_text[:200], "score": s.score}
                    for s in (ctx.sources or [])
                ],
                "confidence": ctx.confidence,
                "is_fallback": ctx.is_fallback,
                "fallback_level": ctx.fallback_level.value,
            })
            return

        # 5. 生成
        try:
            ctx = await self.generation_layer.generate(ctx)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("流式生成异常")
            ctx = self.fallback.no_answer(ctx)

        self._record_elapsed(ctx, t0)
        await self._save_to_session(session_id, original_query, ctx.answer)

        yield SSEEvent("done", {
            "answer": ctx.answer,
            "sources": [
                {"doc_id": s.doc_id, "doc_title": s.doc_title,
                 "chunk_text": s.chunk_text[:200], "score": s.score}
                for s in (ctx.sources or [])
            ],
            "confidence": ctx.confidence,
        })

    @staticmethod
    def _record_elapsed(ctx: PipelineContext, t0: float) -> None:
        """将 pipeline 总耗时（ms）写入 ctx.metadata"""
        ctx.metadata["pipeline_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

    async def _save_to_session(
        self, session_id: str | None, query: str, answer: str
    ) -> None:
        """保存本轮对话到会话历史（同步 SQLite 写入移出事件循环线程）"""
        if not session_id:
            return
        try:
            await asyncio.to_thread(
                self._session_manager.add_message, session_id, "user", query
            )
            await asyncio.to_thread(
                self._session_manager.add_message, session_id, "assistant", answer
            )
        except Exception:
            logger.warning("会话记录失败 session=%s", session_id, exc_info=True)
