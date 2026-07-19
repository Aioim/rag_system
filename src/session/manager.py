"""会话管理器 — TTL / 话题切换 / 上下文压缩"""

import time
import uuid

from config import settings
from logger import logger
from models.session import Message, Session
from session.store import SessionStore, TransactionScope

# 话题切换后保留的消息数（2 轮对话 = user/assistant 各 2 条）
_TOPIC_SWITCH_KEEP_MESSAGES = 4

_VALID_ROLES = frozenset(("user", "assistant", "system"))


def _estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数。

    CJK 及全角字符（码点 >= U+3000）按 2 token/字符的保守上界计，
    其余字符按 4 字符 ≈ 1 token 计。如需精确计数，建议接入对应
    LLM 的 tokenizer。
    """
    wide = sum(1 for ch in text if ord(ch) >= 0x3000)
    return wide * 2 + (len(text) - wide + 3) // 4


class SessionManager:
    """会话生命周期管理器

    功能：
    - 创建/获取/删除会话
    - TTL 自动过期清理
    - 话题切换检测（基于 embedding 余弦相似度）
    - 上下文压缩（token 超限时摘要早期消息）
    """

    def __init__(self, store: SessionStore | None = None):
        self.store = store or SessionStore()
        self._last_cleanup = 0.0

    # ---- 公共 API ----

    def close(self) -> None:
        """关闭底层存储连接（幂等）"""
        self.store.close()

    def get_or_create(self, session_id: str | None = None) -> Session:
        """获取已有会话或创建新会话"""
        self._cleanup_if_needed()
        sid = session_id if session_id is not None else str(uuid.uuid4())
        return self.store.get_or_create(sid)

    def get(self, session_id: str) -> Session | None:
        self._cleanup_if_needed()
        return self.store.get(session_id)

    def add_message(
        self, session_id: str, role: str, content: str,
        embedding: list[float] | None = None,
    ) -> Message:
        """添加消息到会话（原子操作：get→add→update 在同一锁+事务内）

        Args:
            session_id: 会话 ID
            role: user / assistant / system
            content: 消息文本
            embedding: 消息的 embedding 向量（话题检测用）

        Raises:
            ValueError: role 非法或会话不存在
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"非法 role: {role}，应为 {sorted(_VALID_ROLES)} 之一")

        with self.store.transaction() as tx:
            session = tx.get(session_id)
            if session is None:
                raise ValueError(f"会话不存在: {session_id}")

            # 话题切换检测
            if (
                embedding
                and session.topic_embedding
                and self._detect_topic_switch(embedding, session.topic_embedding)
            ):
                logger.info(
                    "话题切换检测: session=%s, 旧话题=%s",
                    session_id, session.current_topic,
                )
                session = self._handle_topic_switch(session, tx, session_id)

            # 更新话题 embedding
            if embedding:
                session.topic_embedding = embedding

            # 添加消息（INSERT + UPDATE last_active）
            msg = tx.add_message(session_id, role, content)
            session.messages.append(msg)

            # 上下文压缩（旧消息归档 + 摘要更新）
            session = self._compress_if_needed(session, tx, session_id)

            # 持久化会话元数据
            tx.update(session)
            return msg

    def delete(self, session_id: str) -> bool:
        return self.store.delete(session_id)

    def get_context(self, session_id: str, max_tokens: int | None = None) -> dict:
        """获取 RAG Pipeline 可用的会话上下文

        Returns:
            {"messages": [...], "summary": str|None, "topic": str|None}
        """
        session = self.store.get(session_id)
        if session is None:
            return {"messages": [], "summary": None, "topic": None}

        if max_tokens is None:
            max_tokens = settings.session.max_context_tokens
        messages = session.messages

        # 简单截断：保留最近的消息直到不超过 max_tokens
        # 始终至少保留最新的一条消息
        selected: list[Message] = []
        token_count = 0
        for msg in reversed(messages):
            msg_tokens = _estimate_tokens(msg.content)
            if selected and token_count + msg_tokens > max_tokens:
                break
            selected.append(msg)
            token_count += msg_tokens
        selected.reverse()

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
        if len(new_embedding) != len(current_embedding):
            # 不抛异常：维度不匹配（如模型版本变更）不应回滚 add_message 事务
            logger.warning(
                "Embedding 维度不匹配，跳过话题切换检测: new=%d, current=%d",
                len(new_embedding), len(current_embedding),
            )
            return False
        threshold = settings.session.topic_switch_threshold
        dot = sum(a * b for a, b in zip(new_embedding, current_embedding, strict=True))
        norm_a = sum(a * a for a in new_embedding) ** 0.5
        norm_b = sum(b * b for b in current_embedding) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return False
        similarity = dot / (norm_a * norm_b)
        return similarity < threshold

    def _handle_topic_switch(
        self, session: Session, tx: TransactionScope, session_id: str
    ) -> Session:
        """话题切换：保留最近 2 轮（4 条），旧消息归档（软删除，可溯源）"""
        session.context_summary = None
        if len(session.messages) > _TOPIC_SWITCH_KEEP_MESSAGES:
            tx.archive_messages_keep_last(session_id, _TOPIC_SWITCH_KEEP_MESSAGES)
            session.messages = session.messages[-_TOPIC_SWITCH_KEEP_MESSAGES:]
        return session

    def _compress_if_needed(
        self, session: Session, tx: TransactionScope, session_id: str
    ) -> Session:
        """超过 token 上限时，将早期消息转为摘要并归档（软删除，可溯源）"""
        max_tokens = settings.session.max_context_tokens
        total = sum(_estimate_tokens(m.content) for m in session.messages)
        if total <= max_tokens:
            return session

        # 从末尾向前累计，找到保留 50% token 预算的切分点
        keep_tokens = 0
        can_compress = False
        keep_from = 0
        for i, msg in enumerate(reversed(session.messages)):
            keep_tokens += _estimate_tokens(msg.content)
            if keep_tokens >= max_tokens * 0.5:
                can_compress = True
                keep_from = len(session.messages) - i - 1
                break

        if can_compress and keep_from > 0:
            # TODO: 调用 LLM 生成实际摘要，替代当前占位文本
            session.context_summary = (
                f"（之前 {keep_from} 条消息因长度限制已省略）"
            )
            keep_count = len(session.messages) - keep_from
            tx.archive_messages_keep_last(session_id, keep_count)
            session.messages = session.messages[keep_from:]
        else:
            # 切分点落在最早一条消息之前（如单条消息即超预算），无法通过
            # 删除早期消息压缩；由 get_context 的截断逻辑兜底
            # 注：措辞避免 "token=" 字样，防止日志脱敏引擎误伤格式占位符
            logger.warning(
                "上下文压缩未生效: session=%s 估算 %d 超出预算 %d，无可删除的早期消息",
                session_id, total, max_tokens,
            )

        return session

    def _cleanup_if_needed(self) -> None:
        """按可配置间隔触发 TTL 清理"""
        now = time.monotonic()
        interval = settings.session.cleanup_interval_seconds
        if now - self._last_cleanup > interval:
            self.store.cleanup_expired()
            self._last_cleanup = now
