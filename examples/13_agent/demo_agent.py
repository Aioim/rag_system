"""
demo_agent.py — ReAct Agent 模块演示

演示内容：
  1. SearchTool / WebSearchTool 工具封装
  2. ToolResult 数据结构
  3. parse_react_output — 解析 LLM 输出
  4. ReActAgent — 思考→行动→观察 循环
  5. AgentResult — 代理执行结果
  6. SSEEvent — 流式输出事件
  7. 重复检测机制

运行方式：
  cd rag0709
  python examples/13_agent/demo_agent.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402, F401
_ = settings.env  # 强制触发 initialize()  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main():
    from types import SimpleNamespace

    # ── 1. 工具结果 (ToolResult) ────────────────────────────────
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

    # ── 2. parse_react_output ───────────────────────────────────
    banner("2. parse_react_output — 解析 ReAct LLM 输出")

    from agent.react_agent import parse_react_output, ParseResult, ParseError

    test_outputs = [
        # 搜索行动
        "THOUGHT: 需要检索年假相关信息\nACTION: search\nQUERY: 年假申请",
        # 完成行动
        "THOUGHT: 信息充分\nACTION: finish\nQUERY: null",
        # 联网搜索
        "THOUGHT: 本地知识库无结果\nACTION: web_search\nQUERY: 最新年假政策",
        # 解析失败
        "无效输出",
    ]

    for i, output in enumerate(test_outputs, 1):
        parsed = parse_react_output(output)
        if isinstance(parsed, ParseResult):
            print(f"  [{i}] ✅ thought={parsed.thought[:40]}... action={parsed.action} query={parsed.query}")
        else:
            print(f"  [{i}] ❌ ParseError: {parsed.error}")

    # ── 3. SearchTool — 知识库搜索工具 ─────────────────────────
    banner("3. SearchTool — 内部知识库搜索")

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

    # ── 4. WebSearchTool — 联网搜索工具 ────────────────────────
    banner("4. WebSearchTool — 联网搜索工具")

    from agent.tools import WebSearchTool

    class MockWebSearcher:
        async def search(self, query: str):
            return f"关于'{query}'的搜索结果: 最新政策..."

    web_tool = WebSearchTool(MockWebSearcher())
    web_result = await web_tool.run("2024年最新年假政策")
    print(f"  搜索 '2024年最新年假政策':")
    print(f"  tool={web_result.tool}, {web_result.elapsed_ms:.1f}ms")
    print(f"  内容: {web_result.content[:80]}...")

    # ── 5. ReActAgent 完整循环 ─────────────────────────────────
    banner("5. ReActAgent — 思考→行动→观察 循环")

    class MockLLM:
        def __init__(self):
            self.round = 0

        async def ainvoke(self, prompt: str, **kwargs):
            self.round += 1
            if self.round == 1:
                return SimpleNamespace(content="THOUGHT: 需要搜索年假信息\nACTION: search\nQUERY: 年假申请流程")
            else:
                return SimpleNamespace(content="THOUGHT: 信息充分，可以回答\nACTION: finish\nQUERY: null")

    from agent.react_agent import ReActAgent
    from config import settings

    agent = ReActAgent(
        llm=MockLLM(),
        search_tool=search_tool,
        web_search_tool=web_tool,
    )

    result = await agent.run("年假怎么申请？", "default")
    print(f"  查询: '年假怎么申请？'")
    print(f"  total_iterations: {result.total_iterations}")
    print(f"  total_elapsed_ms: {result.total_elapsed_ms:.1f}")
    print(f"  react_traces: {len(result.react_traces)} 步")
    if result.react_traces:
        for i, trace in enumerate(result.react_traces, 1):
            print(f"    Step {i}: action={trace.action}, thought={trace.thought[:50]}...")

    # ── 6. AgentResult 结构 ────────────────────────────────────
    banner("6. AgentResult — 执行结果结构")

    from agent.react_agent import AgentResult
    print(f"  AgentResult 字段:")
    print(f"    reranked:         最终检索结果 (list[Chunk])")
    print(f"    react_traces:     每步追踪 (list[ReActTrace])")
    print(f"    total_iterations: 实际迭代次数")
    print(f"    total_elapsed_ms: 总耗时 (ms)")

    # ── 7. SSEEvent — 流式输出 ─────────────────────────────────
    banner("7. SSEEvent — 流式输出事件")

    from agent.react_agent import SSEEvent

    events = [
        SSEEvent(event="react_start", data={"query": "年假怎么申请？"}),
        SSEEvent(event="thought", data={"content": "正在思考..."}),
        SSEEvent(event="action", data={"action": "search", "query": "年假"}),
        SSEEvent(event="observation", data={"results": 3}),
        SSEEvent(event="react_end", data={"iterations": 2}),
    ]

    print("  流式事件序列:")
    for evt in events:
        print(f"    [{evt.event}] {str(evt.data)[:60]}")

    # ── 8. 重复检测机制 ─────────────────────────────────────────
    banner("8. 重复检测机制")

    print(f"  max_consecutive_duplicates: {settings.agent.max_consecutive_duplicates}")
    print(f"  max_iterations:             {settings.agent.max_iterations}")
    print("  机制: 连续输出相同 Action+Query 达阈值时终止循环")
    print("  目的: 防止 Agent 陷入死循环")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ ReAct Agent 模块演示完成")
    print()
    print("  编程接口:")
    print("    from agent import get_react_agent")
    print("    agent = get_react_agent(llm)")
    print("    result = await agent.run(query, collection)")
    print()
    print("  思考循环:")
    print("    Thought → Action → Observation → Thought → ... → Finish")
    print()
    print("  支持的行动:")
    print("    search     — 搜索内部知识库")
    print("    web_search — 联网搜索")
    print("    finish     — 输出最终结果")


if __name__ == "__main__":
    asyncio.run(main())
