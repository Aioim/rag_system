"""EmbedderStage — 批量 embedding，将向量写回 chunk"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from ingestion.context import PipelineContext
from models.enums import DocumentStatus

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbedderStage:
    """使用 SentenceTransformer 对 chunk 文本批量编码"""

    name = "embedder"
    fatal = False

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        from config import settings

        chunks = ctx.chunks
        if not chunks:
            return ctx

        # 过滤已 embedding 的 chunk（幂等）
        pending = [c for c in chunks if c.embedding is None]
        if not pending:
            return ctx

        batch_size = settings.embedding.batch_size
        total_batches = 0
        t0 = time.perf_counter()

        loop = asyncio.get_running_loop()
        for i in range(0, len(pending), batch_size):
            batch = pending[i: i + batch_size]
            texts = [c.text for c in batch]
            # 将同步 encode 提交到线程池避免阻塞 asyncio 事件循环
            embeddings = await loop.run_in_executor(
                None, self.embedding_model.encode, texts
            )
            for c, emb in zip(batch, embeddings, strict=True):
                c.embedding = emb.tolist()
            total_batches += 1

        ctx.metadata["embedding_batches"] = total_batches
        ctx.metadata["embedding_duration_ms"] = (
            time.perf_counter() - t0
        ) * 1000
        ctx.document.status = DocumentStatus.EMBEDDING

        return ctx
