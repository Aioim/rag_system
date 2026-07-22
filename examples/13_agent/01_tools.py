"""
01_tools.py — ReAct Agent：工具封装

演示内容：
  1. ToolResult — 工具执行结果数据结构
  2. SearchTool — 知识库搜索工具
  3. WebSearchTool — 联网搜索工具

运行方式：
  cd rag0709
  python examples/13_agent/01_tools.py

前置条件：需在 .env 中配置 LLM_API_KEY（SearchTool 内部检索需要）
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main():
    # ── 1. ToolResult 数据结构 ──────────────────────────────────
    banner("1. ToolResult — 工具执行结果")

    from agent.tools import ToolResult

    result = ToolResult(
        tool="search",
        query="年假申请流程",
        content="检索到 3 条相关文档分块:\n1. 年假申请流程...\n2. 年假天数规定...",
        chunk_count=3,
        elapsed_ms=45.2,
    )
    print(f"  tool:        {result.tool}")
    print(f"  query:       {result.query}")
    print(f"  chunk_count: {result.chunk_count}")
    print(f"  elapsed_ms:  {result.elapsed_ms:.1f}")
    print(f"  content:     {result.content[:60]}...")

    # ── 2. SearchTool — 知识库搜索 ──────────────────────────────
    banner("2. SearchTool — 内部知识库搜索")

    from agent.tools import SearchTool

    class MockRetrievalLayer:
        async def retrieve(self, ctx, top_k=None):
            from models.chunk import Chunk
            from models.enums import RetrievalEval
            ctx.candidates = [
                Chunk(chunk_id="c1", doc_id="d1",
                      text="年假申请需登录OA填写申请单，经主管审批。",
                      chunk_index=0, rerank_score=0.95),
                Chunk(chunk_id="c2", doc_id="d1",
                      text="工作满1年可享5天带薪年假。",
                      chunk_index=1, rerank_score=0.90),
            ]
            ctx.reranked = ctx.candidates
            ctx.retrieval_eval = RetrievalEval.SUFFICIENT
            return ctx

    search_tool = SearchTool(MockRetrievalLayer())
    result = await search_tool.run("年假申请", "default")
    print(f"  搜索 '年假申请':")
    print(f"  tool={result.tool}, chunks={result.chunk_count}, {result.elapsed_ms:.1f}ms")
    print(f"  内容: {result.content[:80]}...")

    # ── 3. WebSearchTool — 联网搜索工具 ────────────────────────
    banner("3. WebSearchTool — 联网搜索工具")

    from agent.tools import WebSearchTool

    class MockWebSearcher:
        async def search(self, query: str):
            return f"关于'{query}'的搜索结果: 最新政策..."

    web_tool = WebSearchTool(MockWebSearcher())
    web_result = await web_tool.run("2024年最新年假政策")
    print(f"  搜索 '2024年最新年假政策':")
    print(f"  tool={web_result.tool}, {web_result.elapsed_ms:.1f}ms")
    print(f"  内容: {web_result.content[:80]}...")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 工具封装演示完成")
    print()
    print("  下一步: 02_react_loop.py — ReActAgent 思考循环")


if __name__ == "__main__":
    asyncio.run(main())
