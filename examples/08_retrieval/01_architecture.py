"""
01_architecture.py — 检索层：架构概览与配置

演示内容：
  1. RetrievalLayer 架构概览
  2. 检索配置详解
  3. RRF 融合公式与 MMR 多样性公式
  4. Self-RAG 三种评估结果
  5. 组件就绪状态检查

运行方式：
  cd rag0709
  python examples/08_retrieval/01_architecture.py

无需 FAISS 索引即可运行
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings, PROJECT_ROOT  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
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

    # ── 2. 检索配置 ─────────────────────────────────────────────
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

    # ── 3. RRF 与 MMR 公式 ──────────────────────────────────────
    banner("3. RRF 融合与 MMR 多样性")

    print("  RRF 融合公式:")
    print("    RRF_score(d) = Σ 1 / (k + rank_i(d))")
    print(f"    其中 k = {settings.retrieval.rrf_k}")
    print()
    print("  MMR 多样性公式:")
    print("    MMR = argmax[ λ·sim(d,q) - (1-λ)·max_sim(d, already_selected) ]")
    print(f"    其中 λ = {settings.retrieval.mmr_lambda}")
    print()
    print("  Self-RAG 三种评估结果:")
    print("    SUFFICIENT     — 结果充分，直接进入生成")
    print("    NEED_MORE      — 资料不足，触发补充检索(pipeline兜底)")
    print("    INSUFFICIENT   — 完全不足，触发联网搜索/诚实告知")

    # ── 4. 组件就绪状态 ─────────────────────────────────────────
    banner("4. 组件就绪状态")

    from model import models

    faiss_dir = PROJECT_ROOT / settings.faiss.index_dir
    print(f"  FAISS 索引目录:  {faiss_dir}")
    print(f"  索引目录存在:    {faiss_dir.exists()}")
    if faiss_dir.exists():
        collections = [d.name for d in faiss_dir.iterdir() if d.is_dir()]
        print(f"  已有集合:        {collections if collections else '(空)'}")

    emb_status = models.status()
    print(f"  Embedding 模型:  {'✅ 已下载' if emb_status.get('embedding') else '⬜ 未下载'}")
    print(f"  Reranker 模型:   {'✅ 已下载' if emb_status.get('rerank') else '⬜ 未下载'}")

    if not emb_status.get("embedding"):
        print("\n  ⚠️  Embedding 模型未下载，检索无法运行。")
        print("  下载命令: models.download('embedding')")
    if not emb_status.get("rerank"):
        print("\n  ⚠️  Reranker 模型未下载，精排将降级。")
        print("  下载命令: models.download('rerank')")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 检索架构演示完成")
    print()
    print("  下一步: 02_retrieve_demo.py — 实际检索链路（需 FAISS 索引 + Embedding 模型）")
    print()
    print("  前置步骤:")
    print("    1. 下载模型: models.download('embedding')")
    print("    2. 建索引:   python examples/11_ingestion/02_pipeline_index.py")


if __name__ == "__main__":
    main()
