"""
02_edge_cases.py — RAG Pipeline：边界情况与配置

演示内容：
  1. 模糊问题短路
  2. Pipeline 异常独立降级
  3. Fallback 触发场景
  4. 运行统计

运行方式：
  cd rag0709
  python examples/10_core/02_edge_cases.py

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
    import tempfile
    from pathlib import Path

    from examples._llm import create_llm

    # ── 1. 初始化 ───────────────────────────────────────────────
    banner("1. 初始化 RAGPipeline")

    from session.store import SessionStore
    from session.manager import SessionManager
    from core import get_rag_pipeline, reset_rag_pipeline

    llm = create_llm(temperature=0)

    tmp_dir = Path(tempfile.mkdtemp())
    store = SessionStore(db_path=tmp_dir / "demo_edge.db")
    session_manager = SessionManager(store=store)

    pipeline = get_rag_pipeline(llm, session_manager)
    print(f"  ✅ RAGPipeline 已初始化")

    # ── 2. 模糊问题短路 ─────────────────────────────────────────
    banner("2. 模糊问题短路")

    ctx = await pipeline.run("帮帮我")
    print(f"  查询: '帮帮我'")
    print(f"  需要澄清: {ctx.needs_clarification}")
    print(f"  澄清问题: {ctx.clarification_question!r}")
    print(f"  说明: 高模糊问题直接返回澄清请求，不调用检索/生成层")

    # ── 3. Pipeline 异常独立降级 ────────────────────────────────
    banner("3. Pipeline 异常独立降级")

    print("  各层异常处理策略:")
    print("    - 查询理解层失败 → 使用原始 query，意图默认为 concept")
    print("    - 检索层失败     → candidates/reranked 为空，触发兜底")
    print("    - 兜底层失败     → 降级到下一级兜底策略")
    print("    - 生成层失败     → 返回错误提示消息")
    print("    - 会话记录失败   → 仅记录日志，不影响回答返回")
    print()
    print("  ✅ 每层失败时记录日志并继续，不中断 Pipeline")

    # ── 4. Fallback 触发场景 ────────────────────────────────────
    banner("4. Fallback 触发场景与配置")

    print("  三级兜底链路:")
    print("    1️⃣  NEED_MORE    → 补充检索（放宽 top_k）→ PARTIAL")
    print("    2️⃣  INSUFFICIENT → 联网搜索（DuckDuckGo）→ WEB_SEARCH")
    print("    3️⃣  搜索失败     → 诚实告知 → NO_ANSWER")
    print()
    print("  兜底配置:")
    print(f"    max_retrieval_rounds:  {settings.fallback.max_retrieval_rounds}")
    print(f"    web_search.enabled:    {settings.web_search.enabled}")
    print(f"    web_search.provider:   {settings.web_search.provider}")
    print(f"    web_search.timeout:    {settings.web_search.timeout_seconds}s")
    print(f"    no_answer_message:     {settings.fallback.no_answer_message[:40]}...")

    # ── 清理 ────────────────────────────────────────────────────
    reset_rag_pipeline()
    store.close()

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 边界情况演示完成")
    print()
    print("  编程接口:")
    print("    from core import get_rag_pipeline")
    print("    pipeline = get_rag_pipeline(llm, session_manager)")
    print("    ctx = await pipeline.run('查询', session_id='可选')")


if __name__ == "__main__":
    asyncio.run(main())
