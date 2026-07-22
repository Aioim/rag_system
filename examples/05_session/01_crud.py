"""
01_crud.py — 会话管理：会话与消息 CRUD

演示内容：
  1. SessionManager 创建和管理会话
  2. 多轮对话消息添加与读取
  3. 对话历史查询（含归档）
  4. 会话删除

运行方式：
  cd rag0709
  python examples/05_session/01_crud.py
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

    from session.store import SessionStore
    from session.manager import SessionManager

    # 使用临时数据库，不影响现有数据
    tmp_dir = Path(tempfile.mkdtemp())
    db_path = tmp_dir / "demo_sessions.db"

    store = SessionStore(db_path=db_path)
    manager = SessionManager(store=store)

    print(f"  数据库路径: {db_path}")
    print(f"  数据库存在: {db_path.exists()}")

    # ── 1. 创建新会话 ───────────────────────────────────────────
    banner("1. 创建新会话")

    session1 = manager.get_or_create()
    print(f"  自动创建: session_id = {session1.session_id}")
    print(f"  current_topic: {session1.current_topic}")
    print(f"  created_at:    {session1.created_at}")

    # 指定已有 session_id → 返回已有会话
    session1b = manager.get_or_create(session1.session_id)
    print(f"  获取已有: session_id = {session1b.session_id} (同一个)")

    session2 = manager.get_or_create()
    print(f"  第二个会话: session_id = {session2.session_id}")

    # ── 2. 添加对话消息 ─────────────────────────────────────────
    banner("2. 添加多轮对话消息")

    sid = session1.session_id

    msg1 = manager.add_message(sid, "user", "什么是带薪年休假？")
    msg2 = manager.add_message(sid, "assistant", "带薪年休假是员工依法享有的假期，工作满1年可享受5天。")
    print(f"  Round 1:")
    print(f"    [{msg1.role}] {msg1.content}")
    print(f"    [{msg2.role}] {msg2.content}")

    msg3 = manager.add_message(sid, "user", "申请年假需要什么材料？")
    msg4 = manager.add_message(sid, "assistant", "需要填写年假申请单，经主管审批后提交HR。")
    print(f"  Round 2:")
    print(f"    [{msg3.role}] {msg3.content}")
    print(f"    [{msg4.role}] {msg4.content}")

    msg5 = manager.add_message(sid, "user", "年假可以累积到下一年吗？")
    print(f"  Round 3:")
    print(f"    [{msg5.role}] {msg5.content}")

    # ── 3. 获取对话历史 ─────────────────────────────────────────
    banner("3. 获取对话历史")

    messages = store.get_messages(sid)
    print(f"  会话 {sid[:12]}... 共有 {len(messages)} 条消息:")
    for i, m in enumerate(messages, 1):
        print(f"    {i}. [{m.role}] {m.content[:50]}...")

    # 含已归档的完整历史
    full = store.get_messages(sid, include_archived=True)
    print(f"\n  含归档的完整历史: {len(full)} 条")

    # ── 4. 会话删除 ─────────────────────────────────────────────
    banner("4. 会话删除")

    deleted_ok = manager.delete(session2.session_id)
    print(f"  删除 session2: {'✅' if deleted_ok else '❌'}")
    print(f"  再次获取 session2: {manager.get(session2.session_id)}")

    # ── 5. 清理 ─────────────────────────────────────────────────
    store.close()
    db_path.unlink(missing_ok=True)
    print(f"\n  临时数据库已删除: {db_path}")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 会话 CRUD 演示完成")
    print()
    print("  下一步: 02_context_lifecycle.py — 上下文窗口 / 话题切换 / TTL")


if __name__ == "__main__":
    asyncio.run(main())
