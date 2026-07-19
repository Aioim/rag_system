"""PromptAssembler — 上下文组装：去重 → 字符预算分配 → 编号拼接

输入为 reranked 降序结果，Top-1 即最高分。
Lost-in-the-Middle 缓解：Top-1 始终置于组装文本首段，预算不足时截断保留而非丢弃。
"""

from dataclasses import replace

import numpy as np

from config import settings
from models.chunk import Chunk


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))


class PromptAssembler:
    """去重（余弦/文本降级）→ 字符预算 → `[n] 文本` 编号拼接"""

    @staticmethod
    def dedup(chunks: list[Chunk], threshold: float) -> list[Chunk]:
        """与已保留 chunk 相似度 > threshold 时丢弃（输入降序，先到者分高）

        两者均有 embedding 时用余弦相似度，否则降级为文本精确比对。
        另按扩展窗口成员判重：ContextExpander 扩展后 embedding 仍是原始
        向量，无法反映窗口文本重叠；若命中 chunk 的 id 已被某个保留窗口
        覆盖，其正文已在上下文中，直接丢弃。
        """
        kept: list[Chunk] = []
        covered_ids: set[str] = set()
        for c in chunks:
            if c.chunk_id in covered_ids:
                continue
            duplicated = False
            for k in kept:
                if c.embedding is not None and k.embedding is not None:
                    if _cosine_sim(c.embedding, k.embedding) > threshold:
                        duplicated = True
                        break
                elif c.text == k.text:
                    duplicated = True
                    break
            if not duplicated:
                kept.append(c)
                covered_ids.update(
                    c.metadata.get("window_chunk_ids") or [c.chunk_id]
                )
        return kept

    @staticmethod
    def allocate_budget(chunks: list[Chunk], max_chars: int) -> list[Chunk]:
        """按排名顺序装入预算，装不下即停；Top-1 超预算时截断保留"""
        kept: list[Chunk] = []
        used = 0
        for i, c in enumerate(chunks):
            if used + len(c.text) <= max_chars:
                kept.append(c)
                used += len(c.text)
            elif i == 0:
                kept.append(replace(c, text=c.text[:max_chars]))
                used = max_chars
            else:
                break
        return kept

    def assemble(
        self,
        chunks: list[Chunk],
        max_chars: int | None = None,
        threshold: float | None = None,
    ) -> str:
        """完整组装流程；参数缺省时读取 settings.generation"""
        if not chunks:
            return ""
        cfg = settings.generation
        if max_chars is None:
            max_chars = cfg.max_context_chars
        if threshold is None:
            threshold = cfg.dedup_threshold

        deduped = self.dedup(chunks, threshold)
        budgeted = self.allocate_budget(deduped, max_chars)
        return "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(budgeted))
