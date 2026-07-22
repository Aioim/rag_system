"""
02_react_loop.py — ReAct Agent：思考-行动-观察循环

演示内容：
  1. parse_react_output — 解析 LLM 输出
  2. ReActAgent 完整思考循环
  3. AgentResult — 执行结果结构
  4. SSEEvent — 流式输出事件
  5. 重复检测机制

运行方式：
  cd rag0709
  python examples/13_agent/02_react_loop.py

前置条件：需在 .env 中配置 LLM_API_KEY
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
    from examples._llm import create_llm
    from agent.tools import SearchTool, WebSearchTool

    # ── 1. parse_react_output — 解析 ReAct 输出 ────────────────
    banner("1. parse_react_output — 解析 ReAct LLM 输出")

    from agent.react_agent import parse_react_output, ParseResult, ParseError

    test_outputs = [
        "THOUGHT: 需要检索年假相关信息\nACTION: search\nQUERY: 年假申请",
        "THOUGHT: 信息充分\nACTION: finish\nQUERY: null",
        "THOUGHT: 本地知识库无结果\nACTION: web_search\nQUERY: 最新年假政策",
        "无效输出",
    ]

    for i, output in enumerate(test_outputs, 1):
        parsed = parse_react_output(output)
        if isinstance(parsed, ParseResult):
            print(f"  [{i}] ✅ thought={parsed.thought[:40]}... action={parsed.action} query={parsed.query}")
        else:
            print(f"  [{i}] ❌ ParseError: {parsed.error}")

    # ── 2. 准备 Mock 工具 ───────────────────────────────────────
    banner("2. 准备工具")

    class MockRetrievalLayer:
        async def retrieve(self, ctx, top_k=None):
            from models.chunk import Chunk
            from models.enums import RetrievalEval
            ctx.candidates = [
                Chunk(chunk_id="c1", doc_id="d1",
                      text="年假申请需登录OA填写申请单，经主管审批。",
                      chunk_index=0, rerank_score=0.95),
            ]
            ctx.reranked = ctx.candidates
            ctx.retrieval_eval = RetrievalEval.SUFFICIENT
            return ctx

    class MockWebSearcher:
        async def search(self, query: str):
            return f"关于'{query}'的搜索结果: 最新政策..."

    search_tool = SearchTool(MockRetrievalLayer())
    web_tool = WebSearchTool(MockWebSearcher())
    print("  ✅ SearchTool + WebSearchTool 已准备 (Mock 检索层)")

    # ── 3. ReActAgent 完整循环 ─────────────────────────────────
    banner("3. ReActAgent — 思考→行动→观察 循环")

    from agent.react_agent import ReActAgent

    llm = create_llm(temperature=settings.agent.llm_temperature)

    agent = ReActAgent(
        llm=llm,
        search_tool=search_tool,
        web_search_tool=web_tool,
    )

    result = await agent.run("年假怎么申请？", "default")
    print(f"  查询: '年假怎么申请？'")
    print(f"  total_iterations: {result.total_iterations}")
    print(f"  total_elapsed_ms: {result.total_elapsed_ms:.1f}")
    print(f"  react_traces:     {len(result.react_traces)} 步")
    if result.react_traces:
        for i, trace in enumerate(result.react_traces, 1):
            print(f"    Step {i}: action={trace.action}, thought={trace.thought[:50]}...")

    # ── 4. AgentResult 结构 ────────────────────────────────────
    banner("4. AgentResult — 执行结果结构")

    from agent.react_agent import AgentResult
    print(f"  AgentResult 字段:")
    print(f"    reranked:         最终检索结果 (list[Chunk])")
    print(f"    react_traces:     每步追踪 (list[ReActTrace])")
    print(f"    total_iterations: 实际迭代次数")
    print(f"    total_elapsed_ms: 总耗时 (ms)")

    # ── 5. SSEEvent — 流式输出 ─────────────────────────────────
    banner("5. SSEEvent — 流式输出事件")

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

    # ── 6. 重复检测机制 ─────────────────────────────────────────
    banner("6. 重复检测机制")

    print(f"  max_consecutive_duplicates: {settings.agent.max_consecutive_duplicates}")
    print(f"  max_iterations:             {settings.agent.max_iterations}")
    print("  机制: 连续输出相同 Action+Query 达阈值时终止循环")
    print("  目的: 防止 Agent 陷入死循环")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ ReAct Agent 演示完成")
    print()
    print("  思考循环:")
    print("    Thought → Action → Observation → Thought → ... → Finish")


if __name__ == "__main__":
    asyncio.run(main())
