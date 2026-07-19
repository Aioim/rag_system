"""查询改写编排器 — 并行执行 HyDE / 关键词 / 同义词改写，合并去重"""
import asyncio

from logger import logger
from models.llm import LLMProtocol
from query.rewriters.base import BaseRewriter
from query.rewriters.hyde import HyDERewriter
from query.rewriters.keyword_rewriter import KeywordRewriter
from query.rewriters.synonym import SynonymRewriter


class QueryRewriter:
    """查询改写编排器

    并行执行所有注册的 rewriter，合并结果并去重，
    原始 query 始终在返回列表的第一位。
    """

    def __init__(self, llm: LLMProtocol) -> None:
        self._rewriters: list[BaseRewriter] = [
            HyDERewriter(llm, temperature=0.3),
            KeywordRewriter(llm, temperature=0),
            SynonymRewriter(llm, temperature=0.3),
        ]

    async def rewrite(self, query: str) -> list[str]:
        results = await asyncio.gather(
            *(r.rewrite(query) for r in self._rewriters),
            return_exceptions=True,
        )

        all_queries = [query]
        seen = {query}
        for r in results:
            if isinstance(r, (KeyboardInterrupt, SystemExit, asyncio.CancelledError, GeneratorExit)):
                raise r
            # asyncio.gather(return_exceptions=True) 对 Exception 子类返回异常对象，
            # 对 BaseException 的非 Exception 子类（如 CancelledError）直接传播，
            # 因此此处 isinstance(r, BaseException) 实际仅匹配 Exception。
            if isinstance(r, Exception):
                logger.error("QueryRewriter 子改写器异常: %s", r)
                continue
            if not isinstance(r, list):
                logger.warning("QueryRewriter 子改写器返回非 list 类型: %s", type(r).__name__)
                continue
            for q in r:
                if q and q not in seen:
                    seen.add(q)
                    all_queries.append(q)
        return all_queries


# ============================================================================
# 自测：用 Mock Rewriters 演示并行改写编排
# ============================================================================
if __name__ == "__main__":
    import asyncio
    from types import SimpleNamespace

    class _MockRewriter(BaseRewriter):
        """返回固定结果的改写器"""
        def __init__(self, results):
            super().__init__(SimpleNamespace())
            self._results = results

        async def rewrite(self, query: str) -> list[str]:
            return self._results

    async def main():
        rewriters = [
            _MockRewriter(["改写变体A", "改写变体B"]),
            _MockRewriter(["改写变体A"]),  # 重复项，验证去重
            _MockRewriter([]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters

        result = await orchestrator.rewrite("原始查询")
        print("=" * 60)
        print("QueryRewriter 自测")
        print("=" * 60)
        print("  输入: 原始查询")
        print(f"  输出: {result}")
        print(f"  数量: {len(result)} (原始查询 1 + 去重变体 {len(result)-1})")

    asyncio.run(main())
