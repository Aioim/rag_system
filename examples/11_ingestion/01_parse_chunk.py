"""
01_parse_chunk.py — 文档处理：解析器与分块策略

演示内容：
  1. 依赖检查（Embedding 模型下载状态）
  2. 解析器配置（docling / pymupdf4llm / mineru / direct）
  3. 分块策略对比（semantic / fixed / hierarchical）

运行方式：
  cd rag0709
  python examples/11_ingestion/01_parse_chunk.py

前置条件：Embedding 模型已下载
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. 检查 Embedding 模型 ──────────────────────────────────
    banner("1. 检查依赖")

    from model import models

    emb_status = models.status()
    print(f"  Embedding 模型: {'✅ 已下载' if emb_status.get('embedding') else '⬜ 未下载'}")
    if not emb_status.get("embedding"):
        print("  ⚠️  Embedding 模型未下载，请先下载:")
        print("    models.download('embedding')")
        print("  或设置 HUGGINGFACE_TOKEN 环境变量后重试")

    # ── 2. 解析器配置 ───────────────────────────────────────────
    banner("2. 解析器配置")

    print(f"  当前 PDF 解析器:       {settings.ingestion.parsers.get('pdf', 'docling')}")
    print(f"  当前 Markdown 解析器:  {settings.ingestion.parsers.get('md', 'direct')}")
    print(f"  解析后输出目录:        {settings.ingestion.parsed_doc_dir}")
    print()
    print("  可用解析器:")
    print("    docling       — 支持 pdf/docx/pptx/html (默认)")
    print("    pymupdf4llm   — 轻量 PDF 解析")
    print("    mineru        — 高精度 PDF 解析(需额外模型)")
    print("    direct        — 直接读取 md/txt")

    # ── 3. 分块策略 ─────────────────────────────────────────────
    banner("3. 分块策略")

    print(f"  当前策略:    {settings.chunking.strategy}")
    print(f"  chunk_size:  {settings.chunking.chunk_size}")
    print(f"  overlap:     {settings.chunking.overlap}")
    print()
    print("  三种策略对比:")
    print("    semantic      — SentenceTransformer 语义边界切分，通用文档")
    print("    fixed         — 固定窗口 + 滑动步长，结构弱时适用")
    print("    hierarchical  — 按 Markdown 标题层级切分，文档层级清晰时最佳")
    print()
    print("  推荐: 层级清晰的文档用 hierarchical，通用文档用 semantic")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 解析器与分块策略演示完成")
    print()
    print("  下一步: 02_pipeline_index.py — 完整处理 Pipeline + 索引持久化")


if __name__ == "__main__":
    main()
