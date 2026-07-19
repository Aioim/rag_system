"""Chunk 数据模型"""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    context_summary: str | None = None
    embedding: list[float] | None = None
    rerank_score: float = 0.0
    metadata: dict = field(default_factory=dict)
