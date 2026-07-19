"""
安全日志系统包

提供统一日志记录、敏感数据脱敏、性能监控等功能。
主要导出接口：
- LazyLogger：延迟初始化日志实例
- RequestLogger：HTTP请求日志记录器
- log_performance / log_step / log_duration：装饰器和上下文管理器
- mask_sensitive_data：脱敏函数
- SensitiveDataFilter：日志过滤器
"""

from .core import (
    RequestLogger,
    log_duration,
    log_exception,
    log_performance,
    log_security_event,
    log_step,
)
from .filters import SecurityAuditFilter, SensitiveDataFilter
from .lazy import LazyLogger
from .masking import MaskingEngine, mask_sensitive_data

logger = LazyLogger.get("rag")
security_logger = LazyLogger.get("security", separate_log_file="security.log")

__all__ = [
    "LazyLogger",
    "MaskingEngine",
    "RequestLogger",
    "SecurityAuditFilter",
    "SensitiveDataFilter",
    "log_duration",
    "log_exception",
    "log_performance",
    "log_security_event",
    "log_step",
    "logger",
    "mask_sensitive_data",
    "security_logger",
]