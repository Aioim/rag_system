"""
01_single_multi_turn.py — RAG Pipeline：单轮问答与多轮对话

演示内容：
  1. RAGPipeline 初始化
  2. 单轮问答流程
  3. 多轮对话（含会话上下文 + 追问补全）

运行方式：
  cd rag0709
  python examples/10_core/01_single_multi_turn.py

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
    store = SessionStore(db_path=tmp_dir / "demo_core.db")
    session_manager = SessionManager(store=store)

    pipeline = get_rag_pipeline(llm, session_manager)
    print(f"  ✅ RAGPipeline 已初始化")
    print(f"  LLM: {settings.llm.default}")
    print(f"  包含子层: QueryUnderstanding + Retrieval + Generation + Fallback")

    # ── 2. 单轮问答 ─────────────────────────────────────────────
    banner("2. 单轮问答 — 基础流程")

    ctx = await pipeline.run("什么是带薪年休假？")

    print(f"  查询: '什么是带薪年休假？'")
    print(f"  意图: {ctx.intent.value if ctx.intent else 'N/A'}")
    print(f"  回答: {ctx.answer[:120] if ctx.answer else '(空)'}...")
    print(f"  置信度: {ctx.confidence}")
    print(f"  触发兜底: {ctx.is_fallback}")
    print(f"  兜底级别: {ctx.fallback_level.value}")
    print(f"  Pipeline: 查询理解 → 检索 → 兜底检查 → 生成 → 会话记录")

    # ── 3. 多轮对话 ─────────────────────────────────────────────
    banner("3. 多轮对话 — 含会话上下文")

    session = session_manager.get_or_create()
    session_manager.add_message(session.session_id, "user", "带薪年休假怎么申请？")
    session_manager.add_message(
        session.session_id, "assistant",
        "申请带薪年休假需要登录OA系统，填写申请单后提交主管审批。"
    )

    ctx2 = await pipeline.run("需要什么材料？", session_id=session.session_id)

    print(f"  第1轮: '带薪年休假怎么申请？'")
    print(f"  第2轮: '需要什么材料？' (不完整的追问)")
    print(f"  回答: {ctx2.answer[:120] if ctx2.answer else '(空)'}...")
    print(f"  说明: 自动补全为独立查询后再检索生成")

    # ── 清理 ────────────────────────────────────────────────────
    reset_rag_pipeline()
    store.close()

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 单轮与多轮对话演示完成")
    print()
    print("  下一步: 02_edge_cases.py — 模糊问题短路 / 异常降级 / Fallback 场景")


if __name__ == "__main__":
    asyncio.run(main())
