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

from .lazy import LazyLogger
from .core import (
    RequestLogger,
    log_performance,
    log_exception,
    log_security_event,
    log_step,
    log_duration,
)
from .masking import mask_sensitive_data, MaskingEngine
from .filters import SensitiveDataFilter, SecurityAuditFilter
logger = LazyLogger.get("rag")
security_logger = LazyLogger.get("security", separate_log_file="security.log")

__all__ = [
    # 核心日志接口
    "logger",
    "security_logger",
    "LazyLogger",
    "RequestLogger",
    "log_performance",
    "log_exception",
    "log_security_event",
    "log_step",
    "log_duration",
    # 脱敏相关
    "mask_sensitive_data",
    "MaskingEngine",
    "SensitiveDataFilter",
    "SecurityAuditFilter",
]