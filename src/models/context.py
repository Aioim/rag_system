"""Online PipelineContext — QA 链路数据容器"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from models.api import Source
from models.chunk import Chunk
from models.enums import FallbackLevel, Intent, RetrievalEval
from models.session import Session

if TYPE_CHECKING:
    from models.react_trace import ReActTrace


@dataclass
class PipelineContext:
    query: str
    original_query: str = ""
    rewritten_queries: list[str] = field(default_factory=list)
    intent: Intent | None = None
    collection: str = "default"
    candidates: list[Chunk] = field(default_factory=list)
    reranked: list[Chunk] = field(default_factory=list)
    session: Session | None = None
    assembled_prompt: str = ""
    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_eval: RetrievalEval | None = None
    fallback_level: FallbackLevel = FallbackLevel.NONE
    is_fallback: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None
    metadata: dict = field(default_factory=dict)
    react_traces: list[ReActTrace] = field(default_factory=list)
    mode: str = "linear"
    max_iterations: int = 5
