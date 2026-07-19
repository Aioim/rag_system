"""ContextExpander 测试"""
from dataclasses import replace

from models.chunk import Chunk
from retrieval.expander import ContextExpander


def _chunk(i: int, prev_id=None, next_id=None) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}", doc_id="d", text=f"t{i}", chunk_index=i,
        prev_chunk_id=prev_id, next_chunk_id=next_id,
    )


class FakeStore:
    def __init__(self, chunks):
        self._m = {c.chunk_id: c for c in chunks}

    def get_chunk(self, cid):
        c = self._m.get(cid)
        return replace(c, metadata=dict(c.metadata)) if c else None


# 链：c0 <-> c1 <-> c2 <-> c3
CHAIN = [
    _chunk(0, next_id="c1"),
    _chunk(1, prev_id="c0", next_id="c2"),
    _chunk(2, prev_id="c1", next_id="c3"),
    _chunk(3, prev_id="c2"),
]


class TestContextExpander:
    def test_window_1_both_sides(self):
        store = FakeStore(CHAIN)
        c = store.get_chunk("c1")
        result = ContextExpander(store).expand(c, window=1)
        # 返回新 Chunk（不可变），原始 chunk 不变
        assert result is not None
        assert result.text == "t0\nt1\nt2"
        assert result.metadata["window_chunk_ids"] == ["c0", "c1", "c2"]
        # 原始 chunk 未被修改
        assert c.text == "t1"
        assert "window_chunk_ids" not in c.metadata

    def test_window_2(self):
        store = FakeStore(CHAIN)
        c = store.get_chunk("c2")
        result = ContextExpander(store).expand(c, window=2)
        assert result.text == "t0\nt1\nt2\nt3"
        assert c.text == "t2"  # 原始 chunk 不变

    def test_doc_boundary_head(self):
        store = FakeStore(CHAIN)
        c = store.get_chunk("c0")
        result = ContextExpander(store).expand(c, window=1)
        assert result.text == "t0\nt1"
        assert result.metadata["window_chunk_ids"] == ["c0", "c1"]
        assert c.text == "t0"  # 原始 chunk 不变

    def test_missing_neighbor_skipped(self):
        broken = _chunk(9, next_id="ghost")
        store = FakeStore([broken])
        c = store.get_chunk("c9")
        result = ContextExpander(store).expand(c, window=1)
        assert result.text == "t9"
        assert result.metadata["window_chunk_ids"] == ["c9"]
        assert c.text == "t9"  # 原始 chunk 不变

    def test_window_0_noop_text(self):
        store = FakeStore(CHAIN)
        c = store.get_chunk("c1")
        result = ContextExpander(store).expand(c, window=0)
        assert result.text == "t1"
        assert result.metadata["window_chunk_ids"] == ["c1"]
        assert c.text == "t1"  # 原始 chunk 不变
