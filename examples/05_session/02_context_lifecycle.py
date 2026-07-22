"""
02_context_lifecycle.py — 会话管理：上下文窗口与生命周期

演示内容：
  1. 上下文窗口获取（get_context，含自动截断）
  2. 话题切换检测机制
  3. 上下文压缩机制（token 超限自动摘要）
  4. TTL 过期会话清理

运行方式：
  cd rag0709
  python examples/05_session/02_context_lifecycle.py
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

    tmp_dir = Path(tempfile.mkdtemp())
    store = SessionStore(db_path=tmp_dir / "demo_lifecycle.db")
    manager = SessionManager(store=store)

    # 先添加一些对话消息
    session = manager.get_or_create()
    sid = session.session_id
    manager.add_message(sid, "user", "什么是带薪年休假？")
    manager.add_message(sid, "assistant", "带薪年休假是员工依法享有的假期，工作满1年可享受5天。")
    manager.add_message(sid, "user", "申请年假需要什么材料？")
    manager.add_message(sid, "assistant", "需要填写年假申请单，经主管审批后提交HR。")

    # ── 1. 获取上下文窗口 ───────────────────────────────────────
    banner("1. 获取上下文窗口 (get_context)")

    ctx = manager.get_context(sid)
    ctx_msgs = ctx.get("messages", [])
    ctx_summary = ctx.get("summary")
    ctx_topic = ctx.get("topic")

    print(f"  上下文消息数: {len(ctx_msgs)}")
    print(f"  摘要: {ctx_summary or '(暂无)'}")
    print(f"  话题: {ctx_topic or '(暂无)'}")
    print(f"  最大 token 预算: {settings.session.max_context_tokens}")

    # ── 2. 话题切换检测机制 ─────────────────────────────────────
    banner("2. 话题切换检测机制")

    print(f"  话题切换阈值: {settings.session.topic_switch_threshold} (余弦相似度)")
    print(f"  检测方式: 基于消息 embedding 的余弦相似度")
    print(f"  触发行为: 保留最近 2 轮 (4 条)，旧消息归档（软删除，可溯源）")
    print()
    print("  示例: 当前在咨询年假，突然问报销")
    print("    add_message(embedding=报销Embedding)")
    print("    → 检测到相似度 < 阈值 → 自动归档旧消息 → 开始新话题")

    # ── 3. 上下文压缩机制 ───────────────────────────────────────
    banner("3. 上下文压缩机制")

    print(f"  最大上下文 token: {settings.session.max_context_tokens}")
    print(f"  压缩策略: 总 token 超过上限时，将早期消息转为摘要")
    print(f"  保留比例: 约 50% token 预算给最近消息")
    print(f"  归档方式: 软删除（标记 archived=1），可溯源")

    # ── 4. TTL 过期会话清理 ────────────────────────────────────
    banner("4. TTL 过期会话清理")

    print(f"  TTL: {settings.session.ttl_hours} 小时")
    print(f"  清理间隔: {settings.session.cleanup_interval_seconds}s")

    deleted = store.cleanup_expired()
    print(f"  本次清理: {deleted} 个会话（刚刚创建，不会过期）")

    # ── 5. 清理 ─────────────────────────────────────────────────
    store.close()
    db_path.unlink(missing_ok=True)

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 上下文生命周期演示完成")
    print()
    print("  生产环境数据库: data/sessions.db")
    print()
    print("  核心 API:")
    print("    manager.get_context(sid, max_tokens=None) → dict")
    print("    store.cleanup_expired() → int (删除数量)")


if __name__ == "__main__":
    asyncio.run(main())
