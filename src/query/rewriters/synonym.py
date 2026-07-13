"""同义词扩展改写器

用 LLM 生成近义/同义表达变体，扩展查询覆盖面。
"""
from query.rewriters.base import BaseRewriter


class SynonymRewriter(BaseRewriter):
    """生成同义查询变体"""

    def __init__(self, llm):
        self._llm = llm

    async def rewrite(self, query: str) -> list[str]:
        prompt = self._build_prompt(query)
        try:
            response = await self._llm.generate(prompt, temperature=0.3)
            # 按行拆分，过滤空行和与原始查询相同的
            variants = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and line.strip() != query
            ]
            return variants
        except Exception:
            return []

    def _build_prompt(self, query: str) -> str:
        return (
            "为以下查询生成2-3个同义或近义的表达方式，每条一行。"
            "可以换用不同的词汇、句式，但保持原意不变。\n"
            "\n"
            f"原始查询：{query}\n"
            "\n"
            "同义表达："
        )
