"""
03_session_api.py — 数据模型：会话与 API 请求/响应模型

演示内容：
  1. Session / Message — 会话模型
  2. ChatRequest / ChatResponse / SearchRequest — API 模型
  3. Source — 引用来源模型
  4. extract_json_container — JSON 提取工具

运行方式：
  cd rag0709
  python examples/04_models/03_session_api.py
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. Session / Message 模型 ───────────────────────────────
    banner("1. Session / Message — 会话模型")

    from models.session import Session, Message

    session = Session(
        session_id="sess-abc-123",
        current_topic="年假政策咨询",
        context_summary="用户咨询年假相关政策：申请流程、所需材料、天数规定",
        created_at=datetime.datetime.now(),
    )
    print(f"  session_id:       {session.session_id}")
    print(f"  current_topic:    {session.current_topic}")
    print(f"  context_summary:  {session.context_summary[:50]}...")
    print(f"  created_at:       {session.created_at}")

    msg = Message(
        role="user",
        content="年假怎么请？",
    )
    print(f"\n  Message:")
    print(f"    role:    {msg.role}")
    print(f"    content: {msg.content}")

    # ── 2. API 请求/响应模型 ────────────────────────────────────
    banner("2. API 请求/响应模型")

    from models.api import ChatRequest, ChatResponse, SearchRequest, Source

    # 单轮对话请求
    req = ChatRequest(query="什么是带薪年休假？", collection="hr_docs")
    print(f"  ChatRequest.query      = {req.query!r}")
    print(f"  ChatRequest.collection = {req.collection}")

    # 多轮对话请求
    req2 = ChatRequest(query="需要什么材料？", session_id="sess-abc-123", collection="hr_docs")
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
    print(f"\n  ChatResponse.answer     = {resp.answer[:50]}...")
    print(f"  ChatResponse.sources    = {len(resp.sources)} 条")
    print(f"  ChatResponse.confidence = {resp.confidence}")
    print(f"  ChatResponse.session_id = {resp.session_id}")

    # ── 3. Source 模型 ──────────────────────────────────────────
    banner("3. Source — 引用来源模型")

    for i, src in enumerate(sources, 1):
        print(f"  [{i}] doc_id={src.doc_id}, doc_title={src.doc_title}")
        print(f"      chunk_text={src.chunk_text[:50]}...")
        print(f"      score={src.score}")

    # ── 4. extract_json_container ───────────────────────────────
    banner("4. extract_json_container — JSON 提取工具")

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

    # 纯 JSON 和无 JSON
    print(f"  纯 JSON: {extract_json_container('{\"intent\": \"concept\"}')}")
    print(f"  无 JSON: {extract_json_container('无法解析')}")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 会话与 API 模型演示完成")


if __name__ == "__main__":
    main()
