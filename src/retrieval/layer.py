"""RetrievalLayer — 检索层主编排器

召回(向量+BM25 并行) → RRF 融合去重 → 上下文扩展 → 精排+MMR → Self-RAG 自评
"""
import asyncio
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

from logger import logger
from models.chunk import Chunk
from models.context import PipelineContext
from models.enums import RetrievalEval
from retrieval.bm25_retriever import BM25Retriever
from retrieval.evaluator import evaluate
from retrieval.expander import ContextExpander
from retrieval.fusion import rrf_fuse
from retrieval.reranker import Reranker, load_cross_encoder, mmr_select
from retrieval.store import FAISSStore, get_store
from retrieval.vector_retriever import VectorRetriever, load_embedding_model

# 不应被吞没的关键异常（模块级常量，避免每次 retrieve 重新分配）
_CRITICAL_EXC = (KeyboardInterrupt, SystemExit, asyncio.CancelledError, GeneratorExit)


class RetrievalLayer:
    """encoder/cross_encoder 为 None 时首次 retrieve 懒加载真实模型（测试注入 mock）"""

    def __init__(self, encoder: "SentenceTransformer | None" = None, cross_encoder: "CrossEncoder | None" = None):
        self._encoder = encoder
        self._cross_encoder = cross_encoder
        self._bm25_cache: dict[str, BM25Retriever] = {}
        self._bm25_lock = threading.Lock()
        self._encoder_lock = threading.Lock()
        self._cross_encoder_lock = threading.Lock()

    # ---- 懒加载 --------------------------------------------------------

    def _get_encoder(self) -> "SentenceTransformer":
        if self._encoder is not None:
            return self._encoder
        with self._encoder_lock:
            if self._encoder is None:
                self._encoder = load_embedding_model()
            return self._encoder

    def _get_cross_encoder(self) -> "CrossEncoder":
        if self._cross_encoder is not None:
            return self._cross_encoder
        with self._cross_encoder_lock:
            if self._cross_encoder is None:
                self._cross_encoder = load_cross_encoder()
            return self._cross_encoder

    def _get_bm25(self, store: FAISSStore) -> BM25Retriever:
        """按 collection 缓存；store 热重载（version 变化）后重建"""
        # 快速路径：无锁查缓存（GIL 保证 dict.get 原子性）
        cached = self._bm25_cache.get(store.collection)
        if cached is not None and cached.version == store.version:
            return cached
        # 慢速路径：加锁二次检查后构建
        with self._bm25_lock:
            cached = self._bm25_cache.get(store.collection)
            if cached is None or cached.version != store.version:
                cached = BM25Retriever(store)
                self._bm25_cache[store.collection] = cached
            return cached

    @staticmethod
    def _reconstruct_vectors(store: FAISSStore, reranked: list[Chunk]) -> dict[str, np.ndarray | None]:
        """线程池中执行 reconstruct，避免阻塞事件循环；失败记 None（MMR 按完全多样处理）"""
        vectors: dict[str, np.ndarray | None] = {}
        for c in reranked:
            vec = store.reconstruct(c.chunk_id)
            if vec is None:
                logger.warning("chunk %s 向量重建失败，MMR 相似度按 0 处理", c.chunk_id)
            vectors[c.chunk_id] = vec
        return vectors

    # ---- 主流程 --------------------------------------------------------

    @staticmethod
    def _safe_retrieve(retriever: VectorRetriever | BM25Retriever, query: str, k: int, path: str) -> list[str]:
        """单路召回失败降级为空结果，另一路继续"""
        try:
            return retriever.retrieve(query, k)
        except Exception as e:
            logger.error("召回路 [%s] 失败: %s", path, e)
            return []

    async def retrieve(self, ctx: PipelineContext, top_k: int | None = None) -> PipelineContext:
        from config import settings

        cfg = settings.retrieval
        effective_top_k = top_k if top_k is not None else cfg.top_k

        # store 加载（faiss IO）与 BM25 构建/模型加载均为重活，走线程池
        store = await asyncio.to_thread(get_store, ctx.collection)
        if store.is_empty:
            ctx.candidates, ctx.reranked = [], []
            ctx.retrieval_eval = RetrievalEval.INSUFFICIENT
            return ctx

        # 三者相互独立，并行加载（冷启动时模型加载为秒级重活）
        # return_exceptions=True 确保单点失败不取消其余并行加载
        results = await asyncio.gather(
            asyncio.to_thread(self._get_encoder),
            asyncio.to_thread(self._get_cross_encoder),
            asyncio.to_thread(self._get_bm25, store),
            return_exceptions=True,
        )
        encoder_raw, cross_encoder_raw, bm25_raw = results

        encoder_failed = isinstance(encoder_raw, BaseException)
        cross_encoder_failed = isinstance(cross_encoder_raw, BaseException)
        bm25_failed = isinstance(bm25_raw, BaseException)

        # 重新抛出不应被吞没的关键异常
        if isinstance(encoder_raw, _CRITICAL_EXC):
            raise encoder_raw
        if isinstance(cross_encoder_raw, _CRITICAL_EXC):
            raise cross_encoder_raw
        if isinstance(bm25_raw, _CRITICAL_EXC):
            raise bm25_raw

        if encoder_failed:
            logger.error("Encoder 加载失败，检索不可用: %s", encoder_raw)
            ctx.candidates, ctx.reranked = [], []
            ctx.retrieval_eval = RetrievalEval.INSUFFICIENT
            return ctx
        if cross_encoder_failed:
            logger.error("CrossEncoder 加载失败，将跳过精排: %s", cross_encoder_raw)
        if bm25_failed:
            logger.error("BM25 构建失败，仅使用向量检索: %s", bm25_raw)

        encoder = encoder_raw
        cross_encoder = cross_encoder_raw if not cross_encoder_failed else None
        bm25 = bm25_raw if not bm25_failed else None
        vector = VectorRetriever(store, encoder)

        # 1. 每条 query 并行两路召回，每路 top_k×2
        queries = ctx.rewritten_queries or [ctx.query]
        recall_k = effective_top_k * 2
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        tasks = []
        for q in queries:
            tasks.append(loop.run_in_executor(
                None, self._safe_retrieve, vector, q, recall_k, "vector"))
            if bm25 is not None:
                tasks.append(loop.run_in_executor(
                    None, self._safe_retrieve, bm25, q, recall_k, "bm25"))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ranked_lists = []
        for r in results:
            if isinstance(r, BaseException):
                if isinstance(r, _CRITICAL_EXC):
                    raise r
                # _safe_retrieve 已捕获 Exception，此处兜底其余 BaseException，避免丢弃已成功的召回路
                logger.error("召回任务异常: %s", r)
                r = []
            ranked_lists.append(r)
        ctx.metadata["retrieval_recall_ms"] = (time.perf_counter() - t0) * 1000

        # 2. RRF 融合去重 + 截断 → candidates
        fused = rrf_fuse(ranked_lists, cfg.rrf_k, cfg.max_rerank_candidates)
        candidates = []
        for chunk_id, score in fused:
            c = store.get_chunk(chunk_id)
            if c is None:
                logger.warning("chunk %s 在 docstore 中不存在，已跳过", chunk_id)
                continue
            c.metadata["rrf_score"] = score
            candidates.append(c)
        ctx.candidates = candidates

        # 3. 上下文扩展（docstore 内存读，无需线程池）
        t1 = time.perf_counter()
        expander = ContextExpander(store)
        for i, c in enumerate(candidates):
            candidates[i] = expander.expand(c, cfg.expansion_window)
        ctx.metadata["retrieval_expand_ms"] = (time.perf_counter() - t1) * 1000

        # 4. CrossEncoder 精排（对融合后的标准问法 ctx.query）+ MMR 截断
        t2 = time.perf_counter()
        if cross_encoder is not None:
            reranker = Reranker(cross_encoder)
            reranked = await asyncio.to_thread(
                reranker.rerank, ctx.query, candidates
            )
            vectors = await asyncio.to_thread(
                self._reconstruct_vectors, store, reranked
            )
            ctx.reranked = mmr_select(reranked, vectors, effective_top_k, cfg.mmr_lambda)
        else:
            # CrossEncoder 不可用时，直接用 RRF 融合结果截断
            # 将 RRF 得分映射为 rerank_score 代理值，避免自评恒为 INSUFFICIENT
            ctx.reranked = candidates[:effective_top_k]
            for c in ctx.reranked:
                c.rerank_score = c.metadata.get("rrf_score", 0.0)
        ctx.metadata["retrieval_rerank_ms"] = (time.perf_counter() - t2) * 1000

        # 5. Self-RAG 自评
        ctx.retrieval_eval = evaluate(ctx.reranked)
        return ctx
