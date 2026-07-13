"""HyDE (Hypothetical Document Embedding) 改写器

先让 LLM 生成"假设答案"，用假设答案的 embedding 去检索，
因为假设答案在语义空间中比原始 query 更接近真实文档。
"""
from query.rewriters.base import BaseRewriter


class HyDERewriter(BaseRewriter):
    """生成假设答案作为检索查询"""

    def __init__(self, llm):
        self._llm = llm

    async def rewrite(self, query: str) -> list[str]:
        prompt = self._build_prompt(query)
        try:
            response = await self._llm.generate(prompt, temperature=0.3)
            answer = response.strip()
            return [answer] if answer else []
        except Exception:
            return []

    def _build_prompt(self, query: str) -> str:
        return (
            "你是一个知识库助手。请根据用户问题，生成一段100-200字的假设性答案。"
            "不需要完全准确，只需要合理推测可能的答案内容。\n"
            "\n"
            f"用户问题：{query}\n"
            "\n"
            "假设答案（100-200字）："
        )
