"""Ingestion 模块 — 离线文档处理 Pipeline"""

from pathlib import Path

from model import models

from .pipeline import IngestionPipeline
from .parser import ParserStage
from .chunker import ChunkerStage
from .embedder import EmbedderStage
from .indexer import FAISSIndexWriter


def create_default_pipeline() -> IngestionPipeline:
    """组装默认的 ingestion pipeline

    加载 embedding 模型一次，Chunker（SemanticChunker）和 Embedder 共享同一实例。
    """
    from sentence_transformers import SentenceTransformer

    model_path = models.get_path("embedding")
    if model_path is None:
        raise RuntimeError(
            "Embedding 模型未下载，请先运行 models.download('embedding')"
        )

    embedding_model = SentenceTransformer(str(model_path))

    return IngestionPipeline(
        stages=[
            ParserStage(),
            ChunkerStage(embedding_model=embedding_model),
            EmbedderStage(embedding_model=embedding_model),
        ],
        index_writer=FAISSIndexWriter(),
    )
