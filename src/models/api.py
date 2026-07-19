"""API 请求/响应模型"""
from dataclasses import dataclass, field


@dataclass
class Source:
    doc_id: str
    doc_title: str
    chunk_text: str
    score: float


@dataclass
class ChatRequest:
    query: str
    session_id: str | None = None
    collection: str = "default"
    stream: bool = False
    top_k: int = 5


@dataclass
class ChatResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    session_id: str = ""
    confidence: float = 0.0
    is_fallback: bool = False


@dataclass
class SearchRequest:
    query: str
    collection: str = "default"
    top_k: int = 10


@dataclass
class SearchResponse:
    results: list[Source] = field(default_factory=list)
    search_type: str = "hybrid"
