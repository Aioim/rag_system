"""Chunk 测试"""
from models.chunk import Chunk


class TestChunk:
    def test_minimal(self):
        c = Chunk(chunk_id="c1", doc_id="d1", text="hello", chunk_index=0)
        assert c.rerank_score == 0.0
        assert c.embedding is None
        assert c.metadata == {}

    def test_linked_list(self):
        c = Chunk(chunk_id="c-mid", doc_id="d1", text="mid", chunk_index=1,
                   prev_chunk_id="c-0", next_chunk_id="c-2")
        assert c.prev_chunk_id == "c-0"
        assert c.next_chunk_id == "c-2"

    def test_rerank_score(self):
        c = Chunk(chunk_id="c1", doc_id="d1", text="x", chunk_index=0,
                   rerank_score=0.85)
        assert c.rerank_score == 0.85
