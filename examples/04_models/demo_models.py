"""
demo_models.py — 共享数据模型模块演示

演示内容：
  1. PipelineContext — QA 链路核心数据容器
  2. Chunk — 文档分块模型
  3. Document — 文档模型
  4. Session / Message — 会话模型
  5. 枚举类型 (Intent / RetrievalEval / FallbackLevel / DocumentStatus)
  6. API 请求/响应模型 (ChatRequest / ChatResponse / SearchRequest)
  7. Source — 引用来源模型
  8. extract_json_container — JSON 提取工具

运行方式：
  cd rag0709
  python examples/04_models/demo_models.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    import datetime
    from pathlib import Path

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
    print(f"    query            = {ctx.query!r}")
    print(f"    intent           = {ctx.intent}")
    print(f"    candidates       = {len(ctx.candidates)} 条")
    print(f"    reranked         = {len(ctx.reranked)} 条")
    print(f"    confidence       = {ctx.confidence}")
    print(f"    needs_clarification = {ctx.needs_clarification}")

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
    print(f"    intent           = {ctx.intent.value}")
    print(f"    rewritten_queries= {ctx.rewritten_queries}")
    print(f"    candidates       = {len(ctx.candidates)} 条")
    print(f"    retrieval_eval   = {ctx.retrieval_eval.value}")
    print(f"    answer           = {ctx.answer[:50]}...")
    print(f"    confidence       = {ctx.confidence}")

    # ── 3. Chunk 模型 ───────────────────────────────────────────
    banner("3. Chunk — 文档分块模型")

    chunk = Chunk(
        chunk_id="doc1_chunk_3",
        doc_id="doc_001",
        text="带薪年休假是员工依法享有的假期，工作满1年可享受5天带薪年假。",
        chunk_index=3,
        embedding=[0.1, 0.2, 0.3],  # 实际维度1024，此处仅示例
        rerank_score=0.92,
        metadata={"source": "员工手册", "page": 12, "section": "3.2 休假制度"},
    )
    print(f"  chunk_id:     {chunk.chunk_id}")
    print(f"  doc_id:       {chunk.doc_id}")
    print(f"  text:         {chunk.text[:40]}...")
    print(f"  chunk_index:  {chunk.chunk_index}")
    print(f"  rerank_score: {chunk.rerank_score}")
    print(f"  metadata:     {chunk.metadata}")

    # ── 4. Document 模型 ────────────────────────────────────────
    banner("4. Document — 文档模型")

    from models.document import Document

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
    print(f"  doc_id:      {doc.doc_id}")
    print(f"  source_path: {doc.source_path}")
    print(f"  file_type:   {doc.file_type}")
    print(f"  title:       {doc.title}")
    print(f"  status:      {doc.status.value}")
    print(f"  collection:  {doc.collection}")

    # ── 5. Session / Message 模型 ───────────────────────────────
    banner("5. Session / Message — 会话模型")

    from models.session import Session, Message

    session = Session(
        session_id="sess-abc-123",
        current_topic="年假政策咨询",
        context_summary="用户咨询年假相关政策：申请流程、所需材料、天数规定",
        created_at=datetime.datetime.now(),
    )
    print(f"  session_id:      {session.session_id}")
    print(f"  current_topic:   {session.current_topic}")
    print(f"  context_summary: {session.context_summary[:50]}...")
    print(f"  created_at:      {session.created_at}")

    msg = Message(
        role="user",
        content="年假怎么请？",
    )
    print(f"\n  role:        {msg.role}")
    print(f"  content:     {msg.content}")

    # ── 6. API 请求/响应模型 ────────────────────────────────────
    banner("6. API 请求/响应模型")

    from models.api import ChatRequest, ChatResponse, SearchRequest, Source

    # 单轮对话请求
    req = ChatRequest(
        query="什么是带薪年休假？",
        collection="hr_docs",
    )
    print(f"  ChatRequest.query      = {req.query!r}")
    print(f"  ChatRequest.collection = {req.collection}")

    # 多轮对话请求
    req2 = ChatRequest(
        query="需要什么材料？",
        session_id="sess-abc-123",
        collection="hr_docs",
    )
    print(f"  多轮 ChatRequest.session_id = {req2.session_id}")

    # 响应
    sources = [
        Source(doc_id="d1", doc_title="员工手册_2024.pdf", chunk_text="年假申请需要填写...", score=0.95),
        Source(doc_id="d2", doc_title="请假制度.pdf", chunk_text="提交OA审批...", score=0.88),
    ]
    resp = ChatResponse(
        answer="申请年假需要：1. 填写年假申请单 2. 部门主管审批 3. 提交至HR",
        sources=sources,
        confidence=0.91,
        session_id="sess-abc-123",
    )
    print(f"\n  ChatResponse.answer    = {resp.answer[:50]}...")
    print(f"  ChatResponse.sources   = {len(resp.sources)} 条")
    print(f"  ChatResponse.confidence = {resp.confidence}")

    # ── 7. extract_json_container ───────────────────────────────
    banner("7. extract_json_container — JSON 提取工具")

    from utils.json_utils import extract_json_container

    # LLM 返回带 markdown code block 的 JSON
    llm_output = """
    这是意图分析结果：
    ```json
    {"intent": "procedure", "confidence": 0.92}
    ```
    """
    result = extract_json_container(llm_output)
    print(f"  LLM 输出: {llm_output.strip()[:50]}...")
    print(f"  提取结果: {result}")

    # 纯 JSON
    pure_json = '{"intent": "concept"}'
    result2 = extract_json_container(pure_json)
    print(f"  纯 JSON: {pure_json!r} → {result2}")

    # 无 JSON
    no_json = "无法解析"
    result3 = extract_json_container(no_json)
    print(f"  无 JSON: {no_json!r} → {result3}")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 数据模型模块演示完成")
    print()
    print("  所有模型均使用 dataclass / Pydantic，支持类型安全的数据传递。")


if __name__ == "__main__":
    main()
