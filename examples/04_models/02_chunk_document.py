"""
02_chunk_document.py — 数据模型：Chunk 分块与 Document 文档

演示内容：
  1. Chunk — 文档分块模型（含 embedding / rerank_score / metadata）
  2. Document — 文档模型（含状态 / 集合 / 元数据）

运行方式：
  cd rag0709
  python examples/04_models/02_chunk_document.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. Chunk 模型 ───────────────────────────────────────────
    banner("1. Chunk — 文档分块模型")

    from models.chunk import Chunk

    chunk = Chunk(
        chunk_id="doc1_chunk_3",
        doc_id="doc_001",
        text="带薪年休假是员工依法享有的假期，工作满1年可享受5天带薪年假。",
        chunk_index=3,
        embedding=[0.1, 0.2, 0.3],  # 实际维度1024，此处仅示例
        rerank_score=0.92,
        metadata={"source": "员工手册", "page": 12, "section": "3.2 休假制度"},
    )
    print(f"  chunk_id:      {chunk.chunk_id}")
    print(f"  doc_id:        {chunk.doc_id}")
    print(f"  text:          {chunk.text[:40]}...")
    print(f"  chunk_index:   {chunk.chunk_index}")
    print(f"  rerank_score:  {chunk.rerank_score}")
    print(f"  embedding dim: {len(chunk.embedding) if chunk.embedding else 0}")
    print(f"  metadata:      {chunk.metadata}")

    # ── 2. Document 模型 ────────────────────────────────────────
    banner("2. Document — 文档模型")

    from models.document import Document
    from models.enums import DocumentStatus

    doc = Document(
        doc_id="doc_001",
        source_path=Path("/data/docs/员工手册_2024.pdf"),
        file_type="pdf",
        title="员工手册 2024版",
        status=DocumentStatus.DONE,
        collection="hr_docs",
        raw_text="(文档原始内容...)",
        metadata={"author": "HR部门", "version": "2024-v2"},
    )
    print(f"  doc_id:       {doc.doc_id}")
    print(f"  source_path:  {doc.source_path}")
    print(f"  file_type:    {doc.file_type}")
    print(f"  title:        {doc.title}")
    print(f"  status:       {doc.status.value}")
    print(f"  collection:   {doc.collection}")
    print(f"  metadata:     {doc.metadata}")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ Chunk 与 Document 模型演示完成")
    print()
    print("  下一步: 03_session_api.py — Session / Message / API 模型")


if __name__ == "__main__":
    main()
