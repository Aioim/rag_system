"""延迟初始化日志实例模块"""

import contextlib
import logging
import sys
import threading
from datetime import UTC, datetime
from typing import Any

# 使用模块级别的变量来存储日志实例，这样即使模块被多次导入，这些变量也不会被重新初始化
_module_instances: dict[str, Any] = {}
_module_lock = threading.Lock()


class LazyLogger:
    """延迟初始化日志实例"""

    @classmethod
    def get(cls, name: str, **kwargs):
        """获取或创建日志实例"""
        # 直接在锁的保护下检查和创建日志实例，避免多次创建
        with _module_lock:
            if name not in _module_instances:
                # 导入依赖
                from config import settings

                from .filters import SecurityAuditFilter, SensitiveDataFilter
                from .formatters import SecurityFormatter
                from .handlers import HandlerFactory
                log_config = settings.log

                logger = logging.getLogger(name)

                # 清理旧处理器
                if logger.handlers:
                    for handler in logger.handlers[:]:
                        try:
                            logger.removeHandler(handler)
                            handler.close()
                        except Exception:
                            pass

                level = getattr(logging, (kwargs.get('log_level') or log_config.log_level).upper(), logging.INFO)
                logger.setLevel(level)
                logger.propagate = False

                # 添加控制台处理器
                if kwargs.get('log_to_console', True):
                    handler = logging.StreamHandler(sys.stdout)
                    handler.setLevel(logging.DEBUG)
                    formatter = SecurityFormatter(
                        SecurityFormatter.STANDARD_FORMAT,
                        SecurityFormatter.DATE_FORMAT
                    )
                    handler.setFormatter(formatter)
                    logger.addHandler(handler)

                # 添加文件处理器
                if kwargs.get('log_to_file', True):
                    # 主日志器：文件输出 + 错误日志
                    if name == "rag":
                        logger.addHandler(HandlerFactory.create_handler(
                            "timed", str(log_config.log_dir / log_config.log_file), logging.DEBUG
                        ))
                        # 错误日志（含堆栈）
                        logger.addHandler(HandlerFactory.create_handler(
                            "rotating",
                            "error.log",
                            logging.ERROR,
                            fmt="%(asctime)s %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s\nEXCEPTION: %(exc_info)s",
                            maxBytes=log_config.max_bytes
                        ))
                    # 自定义日志文件
                    elif kwargs.get('separate_log_file'):
                        filename = f"{name}.log" if kwargs.get('separate_log_file') is True else kwargs.get('separate_log_file')
                        logger.addHandler(HandlerFactory.create_handler(
                            "timed", filename, logging.DEBUG
                        ))

                # 应用安全过滤器（先审计后脱敏，确保审计看到原始内容）
                logger.addFilter(SecurityAuditFilter())
                logger.addFilter(SensitiveDataFilter())

                # 启动横幅（仅主日志器打印）
                if name == "rag" and not log_config.quiet:
                    logger.info("=" * 70)
                    logger.info(f"[OK] RAG Logger | Env: {settings.env} | Level: {logging.getLevelName(level)}")
                    logger.info(f"[TIME] UTC: {datetime.now(UTC).isoformat()}")
                    logger.info("=" * 70)

                _module_instances[name] = logger
        return _module_instances[name]

    @classmethod
    def cleanup(cls):
        """清理所有日志实例"""
        with _module_lock:
            for logger in _module_instances.values():
                if hasattr(logger, 'handlers'):
                    for handler in logger.handlers:
                        with contextlib.suppress(Exception):
                            handler.close()
                # 从 logging 模块内部注册表中移除
                if hasattr(logger, 'name'):
                    logging.Logger.manager.loggerDict.pop(logger.name, None)
            _module_instances.clear()