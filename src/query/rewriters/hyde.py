"""HyDE (Hypothetical Document Embedding) 改写器

先让 LLM 生成"假设答案"，用假设答案的 embedding 去检索，
因为假设答案在语义空间中比原始 query 更接近真实文档。
"""
from query.rewriters.base import BaseRewriter


class HyDERewriter(BaseRewriter):
    """生成假设答案作为检索查询"""

    def _build_prompt(self, query: str) -> str:
        return (
            "你是一个知识库助手。请根据用户问题，生成一段100-200字的假设性答案。"
            "不需要完全准确，只需要合理推测可能的答案内容。\n"
            "\n"
            f"用户问题：{query}\n"
            "\n"
            "假设答案（100-200字）："
        )


# ============================================================================
# 自测：展示 HyDE Prompt 构建
# ============================================================================
if __name__ == "__main__":
    from types import SimpleNamespace
    r = HyDERewriter(SimpleNamespace())
    print("=" * 60)
    print("HyDERewriter 自测 — Prompt 预览")
    print("=" * 60)
    print(r._build_prompt("什么是零拷贝技术？"))
    print("\ntemperature = 0.3 (建议在构造 ChatOpenAI 时设置)")
