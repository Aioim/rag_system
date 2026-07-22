"""
02_retrieve_demo.py — 检索层：实际检索链路

演示内容：
  1. 多查询实际检索
  2. 粗召回 / 精排 / Self-RAG 评估结果展示
  3. Top-N 结果预览

运行方式：
  cd rag0709
  python examples/08_retrieval/02_retrieve_demo.py

前置条件：
  1. FAISS 索引已建立（运行过 ingestion）
  2. Embedding 模型已下载
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings, PROJECT_ROOT  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_index_exists() -> bool:
    """检查 FAISS 索引是否存在"""
    faiss_dir = PROJECT_ROOT / settings.faiss.index_dir
    idx_file = faiss_dir / "default" / "index.faiss"
    exists = idx_file.exists()
    if not exists:
        # 也检查 demo_collection
        idx_file = faiss_dir / "demo_collection" / "index.faiss"
        exists = idx_file.exists()
    return exists


async def main():
    if not check_index_exists():
        banner("⚠️ 未检测到 FAISS 索引")
        print()
        print("  请先运行 ingestion 建索引:")
        print("    python examples/11_ingestion/02_pipeline_index.py")
        print()
        print("  然后运行架构概览（无需索引）:")
        print("    python examples/08_retrieval/01_architecture.py")
        return

    from model import models
    emb_status = models.status()
    if not emb_status.get("embedding"):
        print("⚠️  Embedding 模型未下载，无法运行实际检索。")
        print("请先运行: models.download('embedding')")
        return

    from models.context import PipelineContext
    from models.enums import Intent
    from retrieval import get_retrieval_layer

    # ── 实际检索 ────────────────────────────────────────────────
    banner("实际检索演示")

    layer = get_retrieval_layer()

    test_queries = [
        ("什么是年假？", Intent.CONCEPT),
        ("申请年假需要什么材料？", Intent.PROCEDURE),
    ]

    for query, intent in test_queries:
        print(f"\n  查询: {query!r}")

        ctx = PipelineContext(query=query)
        ctx.intent = intent

        try:
            ctx = await layer.retrieve(ctx)

            print(f"  粗召回候选:  {len(ctx.candidates)} 条")
            print(f"  精排结果:    {len(ctx.reranked)} 条")
            print(f"  检索评估:    {ctx.retrieval_eval.value if ctx.retrieval_eval else 'N/A'}")

            if ctx.reranked:
                print(f"\n  Top-{min(3, len(ctx.reranked))} 结果:")
                for i, chunk in enumerate(ctx.reranked[:3], 1):
                    score = getattr(chunk, 'rerank_score', 'N/A')
                    text_preview = chunk.text[:80].replace('\n', ' ')
                    print(f"    {i}. [{score}] {text_preview}...")
        except Exception as e:
            print(f"  ⚠️ 检索失败: {e}")

    banner("✅ 检索演示完成")


if __name__ == "__main__":
    asyncio.run(main())
