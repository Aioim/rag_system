"""Ingestion 模块 — 离线文档处理 Pipeline"""

import threading

from model import models

from .chunker import ChunkerStage
from .embedder import EmbedderStage
from .indexer import FAISSIndexWriter
from .parser import ParserStage
from .pipeline import IngestionPipeline

# 缓存：避免多次调用 create_default_pipeline 时重复加载 GB 级模型
_cached_embedding_model = None
_model_load_lock = threading.Lock()


def create_default_pipeline() -> IngestionPipeline:
    """组装默认的 ingestion pipeline

    加载 embedding 模型一次，Chunker（SemanticChunker）和 Embedder 共享同一实例。
    多次调用复用缓存的模型实例（双检锁保证并发下只加载一次）。
    """
    global _cached_embedding_model

    if _cached_embedding_model is None:
        with _model_load_lock:
            if _cached_embedding_model is None:
                from sentence_transformers import SentenceTransformer

                model_path = models.get_path("embedding")
                if model_path is None:
                    raise RuntimeError(
                        "Embedding 模型未下载，请先运行 models.download('embedding')"
                    )
                _cached_embedding_model = SentenceTransformer(str(model_path))

    return IngestionPipeline(
        stages=[
            ParserStage(),
            ChunkerStage(embedding_model=_cached_embedding_model),
            EmbedderStage(embedding_model=_cached_embedding_model),
        ],
        index_writer=FAISSIndexWriter(),
    )
