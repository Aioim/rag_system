"""会话管理器 — TTL / 话题切换 / 上下文压缩"""

import uuid
from typing import Optional

from config import settings
from logger import logger
from models.session import Message, Session
from session.store import SessionStore


class SessionManager:
    """会话生命周期管理器

    功能：
    - 创建/获取/删除会话
    - TTL 自动过期清理
    - 话题切换检测（基于 embedding 余弦相似度）
    - 上下文压缩（token 超限时摘要早期消息）
    """

    def __init__(self, store: Optional[SessionStore] = None):
        self.store = store or SessionStore()

    # ---- 公共 API ----

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """获取已有会话或创建新会话"""
        sid = session_id or str(uuid.uuid4())
        return self.store.get_or_create(sid)

    def get(self, session_id: str) -> Optional[Session]:
        self._cleanup_if_needed()
        return self.store.get(session_id)

    def add_message(
        self, session_id: str, role: str, content: str,
        embedding: Optional[list[float]] = None,
    ) -> Message:
        """添加消息到会话（自动检测话题切换和上下文压缩）

        Args:
            session_id: 会话 ID
            role: user / assistant / system
            content: 消息文本
            embedding: 消息的 embedding 向量（话题检测用）
        """
        session = self.store.get(session_id)
        if session is None:
            raise ValueError(f"会话不存在: {session_id}")

        # 话题切换检测
        if embedding and session.topic_embedding:
            if self._detect_topic_switch(embedding, session.topic_embedding):
                logger.info(
                    "话题切换检测: session=%s, 旧话题=%s",
                    session_id, session.current_topic,
                )
                self._handle_topic_switch(session)

        # 更新话题 embedding
        if embedding:
            session.topic_embedding = embedding

        # 添加消息
        msg = self.store.add_message(session_id, role, content)
        session.messages.append(msg)

        # 上下文压缩
        self._compress_if_needed(session)

        # 持久化
        self.store.update(session)
        return msg

    def delete(self, session_id: str) -> bool:
        return self.store.delete(session_id)

    def get_context(self, session_id: str, max_tokens: Optional[int] = None) -> dict:
        """获取 RAG Pipeline 可用的会话上下文

        Returns:
            {"messages": [...], "summary": str|None, "topic": str|None}
        """
        session = self.store.get(session_id)
        if session is None:
            return {"messages": [], "summary": None, "topic": None}

        max_tokens = max_tokens or settings.session.max_context_tokens
        messages = session.messages

        # 简单截断：保留最近的消息直到不超过 max_tokens
        selected = []
        token_count = 0
        for msg in reversed(messages):
            msg_tokens = len(msg.content)
            if token_count + msg_tokens > max_tokens:
                break
            selected.insert(0, msg)
            token_count += msg_tokens

        return {
            "messages": selected,
            "summary": session.context_summary,
            "topic": session.current_topic,
        }

    # ---- 内部方法 ----

    def _detect_topic_switch(
        self, new_embedding: list[float], current_embedding: list[float]
    ) -> bool:
        """余弦相似度 < 阈值 → 话题已切换"""
        threshold = settings.session.topic_switch_threshold
        dot = sum(a * b for a, b in zip(new_embedding, current_embedding))
        norm_a = sum(a * a for a in new_embedding) ** 0.5
        norm_b = sum(b * b for b in current_embedding) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return False
        similarity = dot / (norm_a * norm_b)
        return similarity < threshold

    def _handle_topic_switch(self, session: Session) -> None:
        """话题切换：保留最近 2 轮，清除摘要和旧消息"""
        session.context_summary = None
        if len(session.messages) > 4:  # 2 轮 = 4 条消息（user+assistant × 2）
            session.messages = session.messages[-4:]

    def _compress_if_needed(self, session: Session) -> None:
        """超过 token 上限时，将早期消息转为摘要"""
        max_tokens = settings.session.max_context_tokens
        total = sum(len(m.content) for m in session.messages)
        if total <= max_tokens:
            return

        # 简单策略：保留最近 50% token 预算的消息，其余移除
        keep_tokens = 0
        keep_from = len(session.messages)
        for i, msg in enumerate(reversed(session.messages)):
            keep_tokens += len(msg.content)
            if keep_tokens >= max_tokens * 0.5:
                keep_from = len(session.messages) - i - 1
                break

        if keep_from > 0 and not session.context_summary:
            session.context_summary = (
                f"（之前 {keep_from} 条消息因长度限制已省略）"
            )
        if keep_from > 0:
            session.messages = session.messages[keep_from:]

    _last_cleanup = 0.0

    def _cleanup_if_needed(self) -> None:
        """每 10 分钟最多触发一次 TTL 清理"""
        import time
        now = time.monotonic()
        if now - self._last_cleanup > 600:
            self.store.cleanup_expired()
            self._last_cleanup = now
