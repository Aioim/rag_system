"""Ingestion 模块 — 离线文档处理 Pipeline"""

from model import models

from .chunker import ChunkerStage
from .embedder import EmbedderStage
from .indexer import FAISSIndexWriter
from .parser import ParserStage
from .pipeline import IngestionPipeline


def create_default_pipeline() -> IngestionPipeline:
    """组装默认的 ingestion pipeline

    Chunker（SemanticChunker）和 Embedder 共享 models.embedding_model 实例。
    """
    embedding_model = models.embedding_model

    return IngestionPipeline(
        stages=[
            ParserStage(),
            ChunkerStage(embedding_model=embedding_model),
            EmbedderStage(embedding_model=embedding_model),
        ],
        index_writer=FAISSIndexWriter(),
    )
