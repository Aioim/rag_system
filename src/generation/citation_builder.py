"""CitationBuilder — 从 reranked chunks 构建引用来源列表"""

from models.api import Source
from models.chunk import Chunk


class CitationBuilder:
    """chunk → Source 映射；doc_title 缺失时降级为 doc_id"""

    @staticmethod
    def build(reranked: list[Chunk]) -> list[Source]:
        return [
            Source(
                doc_id=c.doc_id,
                doc_title=c.metadata.get("doc_title", c.doc_id),
                chunk_text=c.text,
                score=c.rerank_score,
            )
            for c in reranked
        ]
