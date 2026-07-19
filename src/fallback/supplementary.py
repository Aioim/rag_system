"""补充检索器 — NEED_MORE 时放宽 top_k 重新检索

Save → relax top_k → re-retrieve via parameter → merge dedup → re-evaluate.
不修改全局配置，通过 RetrievalLayer.retrieve(top_k=...) 传递放宽后的值。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings
from logger import logger
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval
from retrieval.evaluator import evaluate

if TYPE_CHECKING:
    from retrieval.layer import RetrievalLayer


class SupplementaryRetriever:
    """补充检索器

    当 Self-RAG 评估为 NEED_MORE 时，放宽检索参数重新检索：
    1. 保存当前精排结果
    2. 以放宽的 top_k 重新执行检索
    3. 合并去重新旧结果
    4. 标记 fallback_level = PARTIAL
    """

    # top_k 放宽倍数
    TOP_K_MULTIPLIER = 2
    # top_k 上限
    TOP_K_MAX = 20

    async def retrieve(
        self, ctx: PipelineContext, retrieval_layer: RetrievalLayer
    ) -> PipelineContext:
        """执行补充检索

        Args:
            ctx: 当前 PipelineContext（retrieval_eval 应为 NEED_MORE）
            retrieval_layer: 检索层实例

        Returns:
            更新后的 PipelineContext（含合并后的 reranked 和新 retrieval_eval）
        """
        # 1. 保存旧结果
        old_reranked = list(ctx.reranked)

        original_top_k = settings.retrieval.top_k
        relaxed_top_k = min(original_top_k * self.TOP_K_MULTIPLIER, self.TOP_K_MAX)

        try:
            # 2. 放宽 top_k 重新检索（通过参数传递，不修改全局配置）
            logger.info(
                "补充检索：top_k %d → %d, query=%.100s",
                original_top_k, relaxed_top_k, ctx.query,
            )
            ctx = await retrieval_layer.retrieve(ctx, top_k=relaxed_top_k)

            # 3. 合并去重：保留旧结果中不在新结果里的
            new_ids = {c.chunk_id for c in ctx.reranked}
            merged_old = False
            for chunk in old_reranked:
                if chunk.chunk_id not in new_ids:
                    ctx.reranked.append(chunk)
                    merged_old = True

            # 4. 合并改变了结果集时重新评估，保证 retrieval_eval 与
            #    reranked 数据一致（未合并时检索层的评估本就对应当前数据）
            if merged_old:
                ctx.retrieval_eval = evaluate(ctx.reranked)

            # 5. 若旧结果存在但评估为 INSUFFICIENT，至少恢复到 NEED_MORE
            if ctx.retrieval_eval is RetrievalEval.INSUFFICIENT and old_reranked:
                ctx.retrieval_eval = RetrievalEval.NEED_MORE

        except Exception:
            logger.exception("补充检索异常，保留原始结果")
            # 恢复旧结果；保持原始评估（可能为 NEED_MORE → 上层仍可进入 web_search 兜底）
            ctx.reranked = old_reranked
            # 不强制覆盖 retrieval_eval，保留异常前的状态

        # 6. 标记兜底级别（补充检索完全解决时不做 PARTIAL 降级）
        if ctx.retrieval_eval is not RetrievalEval.SUFFICIENT:
            ctx.fallback_level = FallbackLevel.PARTIAL
            ctx.is_fallback = True

        return ctx
