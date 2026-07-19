"""日志处理器工厂模块"""

import logging
import threading
from pathlib import Path

from config import settings

from .formatters import SecurityFormatter

_initialized_dirs: set = set()
_dir_lock = threading.Lock()


class HandlerFactory:
    """日志处理器工厂"""

    @classmethod
    def _ensure_log_dir(cls, target_dir: Path | None = None) -> Path:
        """确保日志目录存在"""
        log_dir = (target_dir or settings.log.log_dir).resolve()
        with _dir_lock:
            if log_dir in _initialized_dirs:
                return log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            _initialized_dirs.add(log_dir)
            return log_dir

    @classmethod
    def create_handler(
            cls,
            handler_type: str,
            filename: str,
            level: int,
            fmt: str | None = None,
            datefmt: str | None = None,
            **kwargs
    ) -> logging.Handler:
        """创建日志处理器"""
        log_dir = cls._ensure_log_dir()

        if fmt is None:
            fmt = SecurityFormatter.STANDARD_FORMAT
        if datefmt is None:
            datefmt = SecurityFormatter.DATE_FORMAT

        formatter = SecurityFormatter(fmt, datefmt)

        if handler_type == "timed":
            from logging.handlers import TimedRotatingFileHandler
            handler = TimedRotatingFileHandler(
                filename=str(log_dir / filename),
                when=kwargs.get("when", "midnight"),
                interval=kwargs.get("interval", 1),
                backupCount=settings.log.backup_count,
                encoding="utf-8",
                delay=False
            )
        elif handler_type == "rotating":
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                filename=str(log_dir / filename),
                maxBytes=kwargs.get("maxBytes", settings.log.max_bytes),
                backupCount=kwargs.get("backupCount", 5),
                encoding="utf-8",
                delay=False
            )
        elif handler_type == "console":
            import sys
            handler = logging.StreamHandler(sys.stdout)
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler
