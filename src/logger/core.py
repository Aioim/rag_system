"""
高级功能组件模块
提供 RequestLogger、性能监控装饰器、安全事件记录等功能
"""
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from functools import wraps
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

from config import settings

from .helpers import _get_actual_module_name, _get_caller_info
from .lazy import LazyLogger


class ParsedURL(NamedTuple):
    path: str
    params: dict
    full: str


class RequestLogger:
    """安全增强的HTTP请求日志记录器（精准定位业务代码）"""

    def __init__(self, logger=None):
        self.logger = logger or LazyLogger.get("api")

    def log_request(self, method: str, url: str, **kwargs) -> str:
        filename, func_name, lineno = _get_caller_info(skip_frames=2)
        location_prefix = f"[{filename}:{func_name}:{lineno}] "

        request_id = hashlib.sha256(f"{method}{url}{time.time()}".encode()).hexdigest()[:12]

        parsed = self._parse_url(url)
        params_str = self._format_params(parsed.params, max_len=80)

        log_msg = f"{location_prefix}{method.upper()} {parsed.path}"
        if params_str:
            log_msg += f" (params: {params_str})"

        self.logger.info(log_msg)
        return request_id

    def log_response(self, request_id: str, status_code: int, **kwargs):
        filename, func_name, lineno = _get_caller_info(skip_frames=2)
        location_prefix = f"[{filename}:{func_name}:{lineno}] "

        method = kwargs.get('method', 'UNKNOWN').upper()
        url = kwargs.get('url', '')
        duration_ms = kwargs.get('duration_ms', 0.0)

        parsed = self._parse_url(url)
        status_marker = "[OK]" if 200 <= status_code < 300 else "[FAIL]"

        log_msg = f"{location_prefix}{status_marker} {method} {parsed.path} {status_code} ({duration_ms:.1f}ms)"
        self.logger.info(log_msg)

    @staticmethod
    def _parse_url(url: str) -> 'ParsedURL':
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query) if parsed_url.query else {}
        return ParsedURL(parsed_url.path or '/', params, url)

    @staticmethod
    def _format_params(params: dict, max_len: int = 80) -> str:
        if not params:
            return ""

        parts = []
        for k, v_list in params.items():
            v = v_list[0] if v_list else ""
            if any(s in k.lower() for s in settings.log.SENSITIVE_KEYS):
                parts.append(f"{k}=******")
            else:
                display = v[:20] + "..." if len(v) > 20 else v
                parts.append(f"{k}={display}")

        result = ", ".join(parts)
        return result[:max_len] + "..." if len(result) > max_len else result


def log_performance(
    logger=None,
    level=None,
    threshold_ms: float = 50.0,
    enabled: bool = True,
    mark_slow: bool = True
) -> Callable:
    """性能监控装饰器（统一格式）"""
    if not enabled:
        return lambda func: func

    def decorator(func: Callable) -> Callable:
        import logging as _logging
        log = logger or LazyLogger.get("performance")
        actual_module = _get_actual_module_name(func)
        _level = level if level is not None else _logging.INFO

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            except Exception:
                # 异常仍然抛出，但记录耗时
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                msg = f"{actual_module}.{func.__name__} {duration_ms:.2f}ms"
                if mark_slow and duration_ms >= threshold_ms:
                    msg += " [SLOW]"
                with suppress(Exception):
                    log.log(_level, msg)
        return wrapper
    return decorator


def log_exception(logger_param=None, exc: Exception | None = None, context: str = ""):
    log = logger_param or LazyLogger.get("rag")
    try:
        if exc is None:
            _, exc_value, _ = sys.exc_info()
            if exc_value is None:
                return
            exc = exc_value
            tb = traceback.format_exc()
        else:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        msg = f"Exception in {context}: {exc}" if context else str(exc)
        log.error("%s\nTraceback:\n%s", msg, tb)
    except Exception as e:
        error_msg = f"[ERROR] Error in log_exception: {e}"
        if not settings.log.quiet:
            print(error_msg, file=sys.stderr)


def log_security_event(
    action: str,
    user: str = "unknown",
    resource: str = "",
    status: str = "success",
    details: dict | None = None
) -> bool:
    """安全事件记录（混合格式：前缀 + JSON）"""
    log = LazyLogger.get("security")
    try:
        safe_details = {}
        if details:
            for k, v in details.items():
                if any(s in str(k).lower() for s in settings.log.SENSITIVE_KEYS):
                    safe_details[k] = "******"
                else:
                    safe_details[k] = v

        event = {
            "action": action,
            "user": user,
            "resource": resource,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            "ip": os.getenv("CLIENT_IP", "unknown"),
            "env": settings.env,
            "details": safe_details
        }

        # 使用 json.dumps 保证有效 JSON
        log.info(json.dumps(event, ensure_ascii=False, default=str))
        return True
    except Exception as e:
        LazyLogger.get("rag").error("SECURITY_LOG_WRITE_FAILED | action=%s | error=%s", action, e)
        return False


def log_step(step_name: str, logger_param=None):
    log = logger_param or LazyLogger.get("rag")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log.info("[STEP] Step: %s", step_name)
            try:
                result = func(*args, **kwargs)
                log.info("[OK] Completed: %s", step_name)
                return result
            except Exception as e:
                log.error("[FAIL] Failed: %s | %s", step_name, e)
                raise
        return wrapper
    return decorator


@contextmanager
def log_duration(step_name: str, logger_param=None, threshold_ms: float = 50.0):
    """执行时间跟踪（统一格式，精准定位）"""
    log = logger_param or LazyLogger.get("performance")

    # 获取调用者信息（跳过 contextlib 内部）
    filename, func_name, lineno = _get_caller_info(skip_frames=2)
    full_name = f"{filename}:{func_name}:{lineno}"

    start = time.perf_counter()
    log.info("START %s [%s]", step_name, full_name)

    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        msg = f"END {step_name} {duration_ms:.2f}ms"
        if duration_ms >= threshold_ms:
            msg += " [SLOW]"
        with suppress(Exception):
            log.info(msg)