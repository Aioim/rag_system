"""查询改写编排器 — 并行执行 HyDE / 关键词 / 同义词改写，合并去重"""
import asyncio

from query.rewriters.base import BaseRewriter
from query.rewriters.hyde import HyDERewriter
from query.rewriters.keyword import KeywordRewriter
from query.rewriters.synonym import SynonymRewriter


class QueryRewriter:
    """查询改写编排器

    并行执行所有注册的 rewriter，合并结果并去重，
    原始 query 始终在返回列表的第一位。
    """

    def __init__(self, llm):
        self._rewriters: list[BaseRewriter] = [
            HyDERewriter(llm),
            KeywordRewriter(llm),
            SynonymRewriter(llm),
        ]

    async def rewrite(self, query: str) -> list[str]:
        results = await asyncio.gather(
            *(r.rewrite(query) for r in self._rewriters),
            return_exceptions=True,
        )

        all_queries = [query]
        for r in results:
            if isinstance(r, BaseException):
                continue
            for q in r:
                if q and q not in all_queries:
                    all_queries.append(q)
        return all_queries
