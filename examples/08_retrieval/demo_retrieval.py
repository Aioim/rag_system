"""
demo_retrieval.py — 检索层模块演示

演示内容：
  1. RetrievalLayer 架构概览
  2. FAISS 索引读取与文档库概览
  3. BM25 索引构建与检索
  4. RRF 融合策略
  5. 上下文扩展 (prev/next)
  6. CrossEncoder 精排 + MMR 多样性
  7. Self-RAG 检索质量自评
  8. 完整检索链路

运行方式：
  cd rag0709
  python examples/08_retrieval/demo_retrieval.py

前置条件：
  需要至少运行过一次 ingestion（建好 FAISS 索引）:
    python examples/11_ingestion/demo_ingestion.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings, PROJECT_ROOT  # noqa: E402, F401
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_index_exists() -> bool:
    """检查 FAISS 索引是否存在"""
    faiss_dir = PROJECT_ROOT / settings.faiss.index_dir
    idx_file = faiss_dir / "default" / "index.faiss"
    return idx_file.exists()


async def demo_without_index():
    """无索引时演示架构和概念"""
    # ── 1. 检索层架构概览 ───────────────────────────────────────
    banner("1. RetrievalLayer 架构概览")

    print("  RetrievalLayer 编排流程:")
    print("    1️⃣  两路并行召回")
    print("       ├── VectorRetriever  (FAISS 向量检索)")
    print("       └── BM25Retriever     (jieba 分词 BM25)")
    print("    2️⃣  RRF 融合去重截断 (max_rerank_candidates)")
    print("    3️⃣  ContextExpander (prev/next 上下文扩展)")
    print("    4️⃣  CrossEncoderReranker (精排 + MMR 多样性)")
    print("    5️⃣  SelfRAGEvaluator (检索质量自评)")
    print()
    print("  索引目录结构:")
    print(f"    {settings.faiss.index_dir}/")
    print(f"    └── {{collection}}/")
    print(f"        ├── index.faiss        # FAISS 向量索引")
    print(f"        └── docstore.json       # 文档/分块元数据")
    print()
    print("  BM25 索引: 启动时从 docstore 内存构建（jieba 分词）")
    print()

    # ── 2. 关键配置 ─────────────────────────────────────────────
    banner("2. 检索配置")

    print(f"  top_k (最终返回数):          {settings.retrieval.top_k}")
    print(f"  rrf_k (融合参数):            {settings.retrieval.rrf_k}")
    print(f"  max_rerank_candidates:       {settings.retrieval.max_rerank_candidates}")
    print(f"  mmr_lambda (多样性):        {settings.retrieval.mmr_lambda}")
    print(f"  expansion_window:            {settings.retrieval.expansion_window}")
    print(f"  relevance_sufficient:        {settings.retrieval.relevance_threshold_sufficient}")
    print(f"  relevance_need_more:         {settings.retrieval.relevance_threshold_need_more}")
    print(f"  similarity_dedup_threshold:  {settings.retrieval.similarity_dedup_threshold}")
    print(f"  max_context_tokens:          {settings.retrieval.max_context_tokens}")

    # ── 3. 检索流程模拟 (概念演示) ──────────────────────────────
    banner("3. 检索流程概念演示")

    print("  Self-RAG 三种评估结果:")
    print("    SUFFICIENT     — 结果充分，直接进入生成")
    print("    NEED_MORE       — 资料不足，触发补充检索(pipeline兜底)")
    print("    INSUFFICIENT    — 完全不足，触发联网搜索/诚实告知")
    print()
    print("  RRF 融合公式:")
    print("    RRF_score(d) = Σ 1 / (k + rank_i(d))")
    print(f"    其中 k = {settings.retrieval.rrf_k}")
    print()
    print("  MMR 多样性公式:")
    print("    MMR = argmax[ λ·sim(d,q) - (1-λ)·max_sim(d, already_selected) ]")
    print(f"    其中 λ = {settings.retrieval.mmr_lambda}")

    # ── 4. 各组件独立检查 ───────────────────────────────────────
    banner("4. 组件就绪状态")

    from model import models
    from config import PROJECT_ROOT

    faiss_dir = PROJECT_ROOT / settings.faiss.index_dir
    print(f"  FAISS 索引目录:  {faiss_dir}")
    print(f"  索引目录存在:    {faiss_dir.exists()}")
    if faiss_dir.exists():
        collections = [d.name for d in faiss_dir.iterdir() if d.is_dir()]
        print(f"  已有集合:        {collections if collections else '(空)'}")

    emb_status = models.status()
    print(f"  Embedding 模型:  {'✅ 已下载' if emb_status.get('embedding') else '⬜ 未下载'}")
    print(f"  Reranker 模型:   {'✅ 已下载' if emb_status.get('rerank') else '⬜ 未下载'}")
    print()

    if not emb_status.get("embedding"):
        print("  ⚠️  Embedding 模型未下载，检索无法运行。")
        print("  下载命令: models.download('embedding')")
    if not emb_status.get("rerank"):
        print("  ⚠️  Reranker 模型未下载，精排将降级。")
        print("  下载命令: models.download('rerank')")

    banner("✅ 检索模块架构演示完成")
    print()
    print("  下一步:")
    print("    1. 运行 ingestion: python examples/11_ingestion/demo_ingestion.py")
    print("    2. 再次运行本脚本将展示实际检索效果")
    print()
    print("  编程接口:")
    print("    from retrieval import get_retrieval_layer")
    print("    layer = get_retrieval_layer()")
    print("    ctx = await layer.retrieve(pipeline_context)")


async def demo_with_index():
    """有索引时演示实际检索"""
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

    banner("✅ 检索模块演示完成")


async def main():
    if check_index_exists():
        await demo_with_index()
    else:
        print("=" * 60)
        print("  ⚠️  未检测到 FAISS 索引 — 演示检索架构和概念")
        print("=" * 60)
        await demo_without_index()


if __name__ == "__main__":
    asyncio.run(main())
