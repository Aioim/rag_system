"""EmbedderStage — 批量 embedding，将向量写回 chunk"""

import time

from ingestion.context import PipelineContext


class EmbedderStage:
    """使用 SentenceTransformer 对 chunk 文本批量编码"""

    name = "embedder"
    fatal = False

    def __init__(self, embedding_model):
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

        for i in range(0, len(pending), batch_size):
            batch = pending[i: i + batch_size]
            texts = [c.text for c in batch]
            # TODO: 将同步 encode 提交到线程池避免阻塞 asyncio 事件循环
            embeddings = self.embedding_model.encode(texts)
            for c, emb in zip(batch, embeddings):
                c.embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            total_batches += 1

        ctx.metadata["embedding_batches"] = total_batches
        ctx.metadata["embedding_duration_ms"] = (
            time.perf_counter() - t0
        ) * 1000

        return ctx
