"""RAGPipeline — 核心 Pipeline 主编排器

串联完整 RAG 问答链路：
  查询理解 → 检索 → (检索不足→兜底) → 生成 → 会话记录

各层异常独立降级，不中断 Pipeline。
"""
import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.llm import LLMProtocol

from fallback import get_fallback_handler
from fallback.handler import FallbackHandler
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
        self.fallback: FallbackHandler = get_fallback_handler()

    async def run(
        self,
        query: str,
        session_id: str | None = None,
        collection: str = "default",
    ) -> PipelineContext:
        """执行完整 RAG 问答链路"""
        t0 = time.perf_counter()
        ctx = PipelineContext(query=query, collection=collection)

        # ---- 1. 查询理解 ------------------------------------------------
        try:
            ctx = await self.query_layer.process(query, session_id, collection)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("查询理解层异常，使用原始 query 降级继续")

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
        need_short_circuit = await self._apply_fallback(ctx, t0, query, session_id)
        if need_short_circuit:
            return ctx

        # ---- 4. 生成 ----------------------------------------------------
        try:
            ctx = await self.generation_layer.generate(ctx)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("生成层异常，返回兜底消息")
            self.fallback.no_answer(ctx)

        # ---- 5. 记录 ----------------------------------------------------
        self._record_elapsed(ctx, t0)
        await self._save_to_session(session_id, query, ctx.answer)
        return ctx

    async def _apply_fallback(
        self,
        ctx: PipelineContext,
        t0: float,
        query: str,
        session_id: str | None,
    ) -> bool:
        """兜底处理；返回 True 表示已短路（无需进入生成层）"""
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
                self.fallback.no_answer(ctx)

            if ctx.fallback_level in (FallbackLevel.NO_ANSWER, FallbackLevel.WEB_SEARCH):
                self._record_elapsed(ctx, t0)
                await self._save_to_session(session_id, query, ctx.answer)
                return True

        return False

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
