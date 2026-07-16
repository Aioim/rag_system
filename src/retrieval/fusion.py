"""RRF 融合去重 — score = Σ 1/(rrf_k + rank)"""


def rrf_fuse(
    ranked_lists: list[list[str]], rrf_k: int, limit: int
) -> list[tuple[str, float]]:
    """多路排名列表 → 按 RRF 分数降序去重合并，截断至 limit

    位置即 rank（从 1 计）。融合层按"多路排名列表"设计，
    后续新增召回路（如摘要索引）直接追加列表即可。
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:limit]
