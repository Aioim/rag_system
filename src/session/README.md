# Session 模块 — SQLite 会话管理

## 模块概述

Session 模块提供会话生命周期管理，支持多轮对话上下文追踪。

- **会话管理**：创建/查询/归档会话
- **消息追踪**：用户消息 + 助手回复双向记录
- **上下文压缩**：话题切换检测 + 历史消息软删除（`archived=1`）
- **非阻塞 I/O**：SQLite 读写通过 `asyncio.to_thread()` 异步化

## 文件结构

```
session/
├── __init__.py      # 导出 + get_session_manager 单例工厂
├── manager.py       # SessionManager — 会话管理器
├── store.py         # SessionStore — SQLite 持久化层
└── README.md
```

## 快速开始

```python
from session import get_session_manager

sm = get_session_manager()
db_path = "data/sessions.db"   # 由 settings.session.db_path 决定

# 创建或获取会话
session = sm.get_or_create()                # 新建会话
session = sm.get_or_create("abc-123")       # 获取已有会话

# 添加消息
msg = sm.add_message("abc-123", "user", "什么是RAG？")
msg = sm.add_message("abc-123", "assistant", "RAG 是检索增强生成...")

# 获取对话上下文（含摘要）
ctx = sm.get_context("abc-123")
# → {"messages": [...], "summary": "..."}

# 话题切换
sm.mark_topic_switch("abc-123")

# 归档历史消息（软删除，可溯源）
sm.archive_old_messages("abc-123", keep=10)      # 保留最近 10 轮
full = sm.store.get_messages("abc-123", include_archived=True)
```

## 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `session.ttl_hours` | 2 | 会话过期时间 |
| `session.max_history_rounds` | 10 | 最大历史轮次 |
| `session.max_context_tokens` | 4000 | 上下文 token 上限 |
| `session.db_path` | `data/sessions.db` | SQLite 数据库路径 |
| `session.topic_switch_threshold` | 0.5 | 话题切换检测阈值 |
| `session.cleanup_interval_seconds` | 600 | 过期会话清理间隔 |

## 依赖

```bash
pip install aiosqlite
```
