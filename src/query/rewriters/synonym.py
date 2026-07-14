"""同义词扩展改写器

用 LLM 生成近义/同义表达变体，扩展查询覆盖面。
"""
from query.rewriters.base import BaseRewriter


class SynonymRewriter(BaseRewriter):
    """生成同义查询变体"""

    def _parse_response(self, response: str) -> list[str]:
        """按行拆分，过滤空行"""
        return [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip()
        ]

    def _build_prompt(self, query: str) -> str:
        return (
            "为以下查询生成2-3个同义或近义的表达方式，每条一行。"
            "可以换用不同的词汇、句式，但保持原意不变。\n"
            "\n"
            f"原始查询：{query}\n"
            "\n"
            "同义表达："
        )


# ============================================================================
# 自测：展示同义改写 Prompt 构建 + 响应解析
# ============================================================================
if __name__ == "__main__":
    r = SynonymRewriter(None)
    print("=" * 60)
    print("SynonymRewriter 自测 — Prompt 预览 + 响应解析")
    print("=" * 60)
    print("--- Prompt ---")
    print(r._build_prompt("微服务架构的优势"))
    print("\n--- 响应解析示例 ---")
    mock_response = "微服务架构的优点\n微服务架构好处\n微服务有什么优势"
    parsed = r._parse_response(mock_response)
    for i, v in enumerate(parsed, 1):
        print(f"  [{i}] {v}")
    print(f"\ntemperature = 0.3 (建议在构造 ChatOpenAI 时设置)")
