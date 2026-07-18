"""Generation 测试共享 fixtures"""
from types import SimpleNamespace

import pytest

from models.chunk import Chunk
from models.context import PipelineContext
from models.enums import Intent, RetrievalEval


class MockLLM:
    """可编程 Mock LLM — 记录调用，支持失败注入"""

    def __init__(self, response: str = "这是一个回答", should_fail: bool = False):
        self.response = response
        self.should_fail = should_fail
        self.calls: list[tuple[str, dict]] = []

    async def ainvoke(self, prompt: str, **kwargs) -> SimpleNamespace:
        self.calls.append((prompt, kwargs))
        if self.should_fail:
            raise RuntimeError("LLM timeout")
        return SimpleNamespace(content=self.response)


def make_chunk(
    cid: str,
    text: str,
    score: float = 0.9,
    embedding: list[float] | None = None,
    doc_id: str = "d1",
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        text=text,
        chunk_index=0,
        rerank_score=score,
        embedding=embedding,
        metadata=metadata or {},
    )


@pytest.fixture
def sample_ctx() -> PipelineContext:
    ctx = PipelineContext(query="什么是RAG？")
    ctx.intent = Intent.CONCEPT
    ctx.retrieval_eval = RetrievalEval.SUFFICIENT
    ctx.reranked = [
        make_chunk("c0", "RAG是检索增强生成架构", 0.95, [1.0, 0.0]),
        make_chunk("c1", "RAG包含检索和生成两个阶段", 0.85, [0.9, 0.1]),
    ]
    return ctx
