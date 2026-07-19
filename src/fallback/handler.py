"""完整三级兜底处理器

处理检索质量不足的情况：
  NEED_MORE    → 补充检索（放宽 top_k）
  INSUFFICIENT → 联网搜索 → 诚实告知
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings
from fallback.supplementary import SupplementaryRetriever
from fallback.web_search import WebSearcher
from logger import logger
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval

if TYPE_CHECKING:
    from retrieval.layer import RetrievalLayer


class FallbackHandler:
    """三级兜底处理器

    编排完整的兜底链路：
    1. NEED_MORE → 放宽 top_k 补充检索，标记 PARTIAL
    2. INSUFFICIENT → 联网搜索
       - 成功 → 融合搜索结果，标记 WEB_SEARCH
       - 失败/未启用 → 诚实告知，标记 NO_ANSWER
    """

    def __init__(
        self,
        web_searcher: WebSearcher,
        supplementary: SupplementaryRetriever,
    ):
        """初始化兜底处理器

        Args:
            web_searcher: 联网搜索器实例（必需）
            supplementary: 补充检索器实例（必需）
        """
        self._web_searcher = web_searcher
        self._supplementary = supplementary

    # ---- 主入口 ------------------------------------------------------------

    async def handle(
        self,
        ctx: PipelineContext,
        retrieval_layer: RetrievalLayer | None = None,
    ) -> PipelineContext:
        """处理检索不足

        Args:
            ctx: 当前 PipelineContext
            retrieval_layer: 检索层实例（NEED_MORE 时需要）

        Returns:
            更新后的 PipelineContext
        """
        if ctx.retrieval_eval is RetrievalEval.NEED_MORE:
            return await self._handle_need_more(ctx, retrieval_layer)

        if ctx.retrieval_eval is RetrievalEval.INSUFFICIENT:
            return await self._handle_insufficient(ctx)

        return ctx

    # ---- NEED_MORE ---------------------------------------------------------

    async def _handle_need_more(
        self,
        ctx: PipelineContext,
        retrieval_layer: RetrievalLayer | None,
    ) -> PipelineContext:
        """补充检索"""
        if retrieval_layer is None:
            logger.warning("NEED_MORE 但无 retrieval_layer，标记 PARTIAL 继续")
            ctx.fallback_level = FallbackLevel.PARTIAL
            ctx.is_fallback = True
            return ctx

        return await self._supplementary.retrieve(ctx, retrieval_layer)

    # ---- INSUFFICIENT ------------------------------------------------------

    async def _handle_insufficient(self, ctx: PipelineContext) -> PipelineContext:
        """联网搜索 → 诚实告知"""
        try:
            web_result = await self._web_searcher.search(ctx.query)
        except Exception:
            logger.exception("联网搜索异常，降级为诚实告知")
            return self.no_answer(ctx)

        if web_result:
            ctx.fallback_level = FallbackLevel.WEB_SEARCH
            ctx.is_fallback = True
            ctx.answer = web_result
            # 搜索结果视为中等置信度（非检索结果验证，但优于空白）
            ctx.confidence = ctx.confidence or 0.5
            logger.info("联网搜索成功 query=%.100s", ctx.query)
            return ctx

        return self.no_answer(ctx)

    # ---- 诚实告知 ----------------------------------------------------------

    def no_answer(self, ctx: PipelineContext) -> PipelineContext:
        """诚实告知：无法回答"""
        ctx.fallback_level = FallbackLevel.NO_ANSWER
        ctx.is_fallback = True
        ctx.answer = settings.fallback.no_answer_message
        ctx.confidence = 0.0
        return ctx
