"""
02_fusion_rewrite.py — 查询理解：多轮上下文融合与查询改写

演示内容：
  1. 多轮上下文融合（ContextFuser：指代消解 + 追问补全）
  2. 查询改写（QueryRewriter：HyDE / Keyword / Synonym 并行执行）

运行方式：
  cd rag0709
  python examples/07_query/02_fusion_rewrite.py

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

    # ── 1. 初始化 LLM ───────────────────────────────────────────
    banner("1. 初始化 LLM")
    llm = create_llm(temperature=0)
    print(f"  ✅ LLM 已连接: {settings.llm.default}")

    # ── 2. 多轮上下文融合 ───────────────────────────────────────
    banner("2. 多轮上下文融合 — 指代消解 + 追问补全")

    from query.context_fuser import ContextFuser
    from models.session import Session, Message

    fuser = ContextFuser(llm)
    print("  组件温度: 0 (确定性输出)")

    # 模拟多轮对话的 Session
    session = Session(
        session_id="demo-session",
        messages=[
            Message(role="user", content="带薪年休假怎么申请？"),
            Message(role="assistant", content="申请带薪年休假需要提前在OA系统中提交申请，经主管审批。"),
        ]
    )

    # 追问补全
    fused = await fuser.fuse("需要什么材料？", session)
    print(f"  第1轮: 用户问'带薪年休假怎么申请？'")
    print(f"  第2轮: 用户问'需要什么材料？' (不完整)")
    print(f"  融合后: {fused!r}")
    print(f"  说明: 自动补全为完整的独立查询")

    # 指代消解
    session.messages.append(Message(role="user", content="需要什么材料？"))
    session.messages.append(Message(role="assistant", content="申请年假需要填写OA申请单、提供身份证明。"))
    fused2 = await fuser.fuse("上面提到的流程具体怎么做？", session)
    print(f"\n  第3轮: 用户问'上面提到的流程具体怎么做？'")
    print(f"  融合后: {fused2!r}")

    # ── 3. 查询改写 ─────────────────────────────────────────────
    banner("3. 查询改写 — HyDE / 关键词 / 同义变体")

    from query.rewriters import QueryRewriter

    rewriter = QueryRewriter(llm)
    rewritten = await rewriter.rewrite("什么是RAG？")

    print(f"  原始查询: '什么是RAG？'")
    print(f"  改写结果 ({len(rewritten)} 条):")
    for i, rw in enumerate(rewritten, 1):
        print(f"    {i}. {rw}")
    print()
    print("  改写器说明:")
    print("    - HyDERewriter (t=0.3):   生成假设答案用于检索")
    print("    - KeywordRewriter (t=0):  提取 BM25 关键词（幂等）")
    print("    - SynonymRewriter (t=0.3): 生成同义变体（多样性）")
    print("    - 三个改写器并行执行，合并去重")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 上下文融合与查询改写演示完成")
    print()
    print("  下一步: 03_full_pipeline.py — QueryUnderstandingLayer 完整流程")


if __name__ == "__main__":
    asyncio.run(main())
