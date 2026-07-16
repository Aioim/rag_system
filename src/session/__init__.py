"""会话管理模块"""

import threading

from session.store import SessionStore
from session.manager import SessionManager


_session_manager: SessionManager | None = None
_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """惰性获取全局 SessionManager 单例（线程安全）"""
    global _session_manager
    if _session_manager is None:
        with _lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    return _session_manager


__all__ = ["SessionStore", "SessionManager", "get_session_manager"]
