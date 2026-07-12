"""指标收集模块"""

import threading


class LogMetrics:
    """SRE监控指标"""
    _lock = threading.Lock()
    _stats = {
        "total_logs": 0, "filtered_logs": 0, "masking_time_ns": 0,
        "handler_errors": 0, "password_leak_attempts": 0,
        "serialization_failures": 0,
    }
    @classmethod
    def record(cls, key: str, value: int = 1):
        """记录指标"""
        with cls._lock:
            if key not in cls._stats:
                cls._stats[key] = 0
            cls._stats[key] = cls._stats.get(key, 0) + value

