"""关键词提取改写器

从 query 中提取纯关键词，给 BM25 做稀疏检索。
"""
from query.rewriters.base import BaseRewriter


class KeywordRewriter(BaseRewriter):
    """提取关键词用于 BM25 检索"""

    def _build_prompt(self, query: str) -> str:
        return (
            "从以下用户问题中提取最重要的关键词（名词、动词、专有名词），"
            '用空格分隔，不要包含"什么""如何""怎么"等疑问词。\n'
            "\n"
            f"用户问题：{query}\n"
            "\n"
            "关键词："
        )


# ============================================================================
# 自测：用 LangChain + DeepSeek API 真实提取关键词
# ============================================================================
if __name__ == "__main__":
    import asyncio
    import os

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    async def main():
        api_key = os.getenv("API_KEY", "")
        if not api_key:
            print("❌ 未找到 API_KEY，请检查 .env 文件")
            return

        # 防止误触发真实 API 调用
        if os.getenv("SELF_TEST") != "1":
            print("⚠️  此自测会发起真实 API 调用（api.deepseek.com）。")
            print("   设置环境变量 SELF_TEST=1 以继续。")
            return

        print("=" * 60)
        print("KeywordRewriter 自测 — LangChain + DeepSeek 关键词提取")
        print("=" * 60)

        chat_model = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            temperature=0,
        )

        r = KeywordRewriter(chat_model)
        queries = [
            "如何配置Nginx反向代理？",
            "五险一金缴纳比例是多少？",
            "年假申请需要什么材料？",
        ]

        for q in queries:
            try:
                result = await r.rewrite(q)
                keywords = result[0] if result else "(空)"
                print(f"\n  用户问题: {q}")
                print(f"  提取关键词: {keywords}")
            except Exception as exc:
                print(f"\n  用户问题: {q}")
                print(f"  ❌ 失败: {exc}")

    asyncio.run(main())
