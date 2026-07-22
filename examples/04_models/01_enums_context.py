"""
01_enums_context.py — 数据模型：枚举类型与 PipelineContext

演示内容：
  1. 枚举类型（Intent / RetrievalEval / FallbackLevel / DocumentStatus）
  2. PipelineContext — QA 链路核心数据容器

运行方式：
  cd rag0709
  python examples/04_models/01_enums_context.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. 枚举类型 ─────────────────────────────────────────────
    banner("1. 枚举类型")

    from models.enums import Intent, RetrievalEval, FallbackLevel, DocumentStatus

    print("  Intent (用户意图):")
    for v in Intent:
        print(f"    Intent.{v.name} = '{v.value}'")

    print("  RetrievalEval (检索评估):")
    for v in RetrievalEval:
        print(f"    RetrievalEval.{v.name} = '{v.value}'")

    print("  FallbackLevel (兜底级别):")
    for v in FallbackLevel:
        print(f"    FallbackLevel.{v.name} = '{v.value}'")

    print("  DocumentStatus (文档状态):")
    for v in DocumentStatus:
        print(f"    DocumentStatus.{v.name} = '{v.value}'")

    # ── 2. PipelineContext — 核心容器 ───────────────────────────
    banner("2. PipelineContext — QA 链路核心数据容器")

    from models.context import PipelineContext
    from models.chunk import Chunk

    ctx = PipelineContext(query="什么是RAG？")
    print(f"  初始状态:")
    print(f"    query                = {ctx.query!r}")
    print(f"    intent               = {ctx.intent}")
    print(f"    candidates           = {len(ctx.candidates)} 条")
    print(f"    reranked             = {len(ctx.reranked)} 条")
    print(f"    confidence           = {ctx.confidence}")
    print(f"    needs_clarification  = {ctx.needs_clarification}")

    # 模拟 Pipeline 各阶段填充数据
    ctx.intent = Intent.CONCEPT
    ctx.rewritten_queries = ["什么是检索增强生成", "RAG架构原理"]
    ctx.candidates = [
        Chunk(chunk_id="c1", doc_id="d1", text="RAG是检索增强生成架构...", chunk_index=0),
        Chunk(chunk_id="c2", doc_id="d1", text="RAG包含检索和生成两个阶段...", chunk_index=1),
    ]
    ctx.retrieval_eval = RetrievalEval.SUFFICIENT
    ctx.answer = "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI架构..."
    ctx.confidence = 0.85

    print(f"\n  处理完成后:")
    print(f"    intent               = {ctx.intent.value}")
    print(f"    rewritten_queries    = {ctx.rewritten_queries}")
    print(f"    candidates           = {len(ctx.candidates)} 条")
    print(f"    retrieval_eval       = {ctx.retrieval_eval.value}")
    print(f"    answer               = {ctx.answer[:50]}...")
    print(f"    confidence           = {ctx.confidence}")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 枚举与上下文容器演示完成")
    print()
    print("  下一步: 02_chunk_document.py — Chunk 与 Document 模型")


if __name__ == "__main__":
    main()
