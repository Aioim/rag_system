"""关键词提取改写器

从 query 中提取纯关键词，给 BM25 做稀疏检索。
"""
from query.rewriters.base import BaseRewriter


class KeywordRewriter(BaseRewriter):
    """提取关键词用于 BM25 检索"""

    def __init__(self, llm):
        self._llm = llm

    async def rewrite(self, query: str) -> list[str]:
        prompt = self._build_prompt(query)
        try:
            response = await self._llm.generate(prompt, temperature=0)
            keywords = response.strip()
            return [keywords] if keywords else []
        except Exception:
            return []

    def _build_prompt(self, query: str) -> str:
        return (
            "从以下用户问题中提取最重要的关键词（名词、动词、专有名词），"
            '用空格分隔，不要包含"什么""如何""怎么"等疑问词。\n'
            "\n"
            f"用户问题：{query}\n"
            "\n"
            "关键词："
        )
