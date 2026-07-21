"""
demo_session.py — 会话管理模块演示

演示内容：
  1. SessionManager 创建和管理会话
  2. 多轮对话消息添加与读取
  3. 上下文窗口获取（含自动摘要/截断）
  4. 话题切换检测（需 embedding）
  5. 上下文压缩（token 超限自动摘要）
  6. TTL 过期会话清理

运行方式：
  cd rag0709
  python examples/05_session/demo_session.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 先触发配置初始化，打破循环导入
from config import settings  # noqa: E402, F401
_ = settings.env  # 强制触发 initialize()，在导入 session 前完成 _config 设置  # noqa: E402


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

    # 无 session_id → 自动生成 UUID
    session1 = manager.get_or_create()
    print(f"  自动创建: session_id = {session1.session_id}")
    print(f"  current_topic: {session1.current_topic}")
    print(f"  created_at:    {session1.created_at}")

    # 指定已有 session_id → 返回已有会话
    session1b = manager.get_or_create(session1.session_id)
    print(f"  获取已有: session_id = {session1b.session_id} (同一个)")

    # 创建第二个会话
    session2 = manager.get_or_create()
    print(f"  第二个会话: session_id = {session2.session_id}")

    # ── 2. 添加对话消息 ─────────────────────────────────────────
    banner("2. 添加多轮对话消息")

    sid = session1.session_id

    # 第一轮
    msg1 = manager.add_message(sid, "user", "什么是带薪年休假？")
    msg2 = manager.add_message(sid, "assistant", "带薪年休假是员工依法享有的假期，工作满1年可享受5天。")
    print(f"  Round 1:")
    print(f"    [{msg1.role}] {msg1.content}")
    print(f"    [{msg2.role}] {msg2.content}")

    # 第二轮
    msg3 = manager.add_message(sid, "user", "申请年假需要什么材料？")
    msg4 = manager.add_message(sid, "assistant", "需要填写年假申请单，经主管审批后提交HR。")
    print(f"  Round 2:")
    print(f"    [{msg3.role}] {msg3.content}")
    print(f"    [{msg4.role}] {msg4.content}")

    # 第三轮
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

    # ── 4. 获取上下文窗口（Pipeline 使用） ──────────────────────
    banner("4. 获取上下文窗口（含自动截断）")

    ctx = manager.get_context(sid)
    # get_context 返回 dict: {"messages": [...], "summary": str|None, "topic": str|None}
    ctx_msgs = ctx.get("messages", [])
    ctx_summary = ctx.get("summary")
    ctx_topic = ctx.get("topic")
    print(f"  上下文消息数: {len(ctx_msgs)}")
    print(f"  摘要: {ctx_summary or '(暂无)'}")
    print(f"  话题: {ctx_topic or '(暂无)'}")
    print(f"  最大 token 预算: {settings.session.max_context_tokens}")

    # ── 5. 话题切换检测机制 ─────────────────────────────────────
    banner("5. 话题切换检测机制")

    print(f"  话题切换阈值: {settings.session.topic_switch_threshold} (余弦相似度)")
    print(f"  检测方式: 基于消息 embedding 的余弦相似度")
    print(f"  触发行为: 保留最近 2 轮 (4 条)，旧消息归档（软删除，可溯源）")
    print()
    print("  示例: 当前在咨询年假，突然问报销")
    print("    add_message(embedding=报销Embedding)")
    print("    → 检测到相似度 < 阈值 → 自动归档旧消息 → 开始新话题")

    # ── 6. 上下文压缩机制 ───────────────────────────────────────
    banner("6. 上下文压缩机制")

    print(f"  最大上下文 token: {settings.session.max_context_tokens}")
    print(f"  压缩策略: 总 token 超过上限时，将早期消息转为摘要")
    print(f"  保留比例: 约 50% token 预算给最近消息")
    print(f"  归档方式: 软删除（标记 archived=1），可溯源")

    # ── 7. TTL 过期会话清理 ────────────────────────────────────
    banner("7. TTL 过期会话清理")

    print(f"  TTL: {settings.session.ttl_hours} 小时")
    print(f"  清理间隔: {settings.session.cleanup_interval_seconds}s")

    deleted = store.cleanup_expired()
    print(f"  本次清理: {deleted} 个会话（刚刚创建，不会过期）")

    # ── 8. 会话删除 ─────────────────────────────────────────────
    banner("8. 会话删除")

    deleted_ok = manager.delete(session2.session_id)
    print(f"  删除 session2: {'✅' if deleted_ok else '❌'}")
    print(f"  再次获取 session2: {manager.get(session2.session_id)}")

    # ── 9. 清理临时数据库 ───────────────────────────────────────
    banner("9. 清理")

    store.close()
    db_path.unlink(missing_ok=True)
    print(f"  临时数据库已删除: {db_path}")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 会话管理模块演示完成")
    print()
    print("  生产环境数据库: data/sessions.db (由 config session.db_path 控制)")
    print()
    print("  核心 API:")
    print("    manager.get_or_create(session_id)  → Session")
    print("    manager.add_message(sid, role, content, embedding=None) → Message")
    print("    manager.get_context(sid, max_tokens=None) → dict")
    print("    manager.delete(sid) → bool")
    print("    store.get_messages(sid, include_archived=False) → list[Message]")
    print("    store.cleanup_expired() → int (删除数量)")
    print()
    print("  Message 模型字段: role, content, timestamp, row_id (DB rowid)")
    print("  Session 模型字段: session_id, messages, context_summary, current_topic, created_at")


if __name__ == "__main__":
    asyncio.run(main())
