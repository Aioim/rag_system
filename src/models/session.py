"""Session + Message 数据模型"""
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    row_id: int = 0  # DB rowid; 0 表示尚未持久化


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    context_summary: str | None = None
    current_topic: str | None = None
    topic_embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))
