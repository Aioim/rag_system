"""ContextExpander — 沿 prev/next chunk_id 拼接窗口文本

相邻命中窗口重叠导致的重复文本由 generation 组装层去重，本模块不处理。
"""
from models.chunk import Chunk


class ContextExpander:
    def __init__(self, store):
        self._store = store

    def expand(self, chunk: Chunk, window: int) -> Chunk:
        """向左右各拉 window 个邻居，按 chunk_index 顺序拼接写回 chunk.text"""
        if "window_chunk_ids" in chunk.metadata:  # 已扩展，幂等返回
            return chunk

        before: list[Chunk] = []
        cur = chunk
        for _ in range(window):
            if not cur.prev_chunk_id:
                break
            prev = self._store.get_chunk(cur.prev_chunk_id)
            if prev is None:
                break
            before.insert(0, prev)
            cur = prev

        after: list[Chunk] = []
        cur = chunk
        for _ in range(window):
            if not cur.next_chunk_id:
                break
            nxt = self._store.get_chunk(cur.next_chunk_id)
            if nxt is None:
                break
            after.append(nxt)
            cur = nxt

        window_chunks = [*before, chunk, *after]
        chunk.metadata["window_chunk_ids"] = [c.chunk_id for c in window_chunks]
        chunk.text = "\n".join(c.text for c in window_chunks)
        return chunk
