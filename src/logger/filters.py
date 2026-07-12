"""
日志记录过滤器
提供敏感数据拦截与指标统计功能，可附加到日志处理器
"""
import logging

from config import settings
from .masking import mask_sensitive_data
from .metrics import LogMetrics

LogConfig = settings.log


class SensitiveDataFilter(logging.Filter):
    """
    敏感数据过滤器

    在日志记录被处理器处理之前：
    1. 对消息内容进行脱敏
    2. 对参数字典中的敏感字段值进行遮蔽
    3. 更新监控指标

    使用方法：
        logger.addFilter(SensitiveDataFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        过滤日志记录，执行脱敏操作

        Args:
            record: 日志记录对象

        Returns:
            True（始终允许记录通过，仅修改内容）
        """
        try:
            # 1. 脱敏日志消息
            if isinstance(record.msg, str):
                record.msg = mask_sensitive_data(record.msg)

            # 2. 处理参数字典
            if record.args:
                if isinstance(record.args, dict):
                    # 对字典中的敏感键进行遮蔽
                    sanitized = {}
                    for key, value in record.args.items():
                        if any(s in str(key).lower() for s in LogConfig.SENSITIVE_KEYS):
                            sanitized[key] = "******"
                        elif isinstance(value, str):
                            sanitized[key] = mask_sensitive_data(value)
                        else:
                            sanitized[key] = value
                    record.args = sanitized

                elif isinstance(record.args, (list, tuple)):
                    # 对序列中的字符串元素进行脱敏
                    new_args = []
                    for arg in record.args:
                        if isinstance(arg, str):
                            new_args.append(mask_sensitive_data(arg))
                        elif isinstance(arg, dict):
                            # 对嵌套字典进行深度处理
                            sanitized = {}
                            for k, v in arg.items():
                                if any(s in str(k).lower() for s in LogConfig.SENSITIVE_KEYS):
                                    sanitized[k] = "******"
                                elif isinstance(v, str):
                                    sanitized[k] = mask_sensitive_data(v)
                                else:
                                    sanitized[k] = v
                            new_args.append(sanitized)
                        else:
                            new_args.append(arg)

                    # 保持原类型
                    if isinstance(record.args, tuple):
                        record.args = tuple(new_args)
                    else:
                        record.args = new_args

            # 3. 更新监控指标
            LogMetrics.record("filtered_logs")
            LogMetrics.record("total_logs")

            return True

        except Exception as e:
            # 过滤过程发生异常时，尽量不影响日志记录本身
            LogMetrics.record("handler_errors")
            if not LogConfig.quiet:
                import sys
                print(f"⚠️  SensitiveDataFilter error: {e}", file=sys.stderr)
            return True  # 仍然允许日志通过，只是可能未脱敏


class SecurityAuditFilter(logging.Filter):
    """
    安全审计过滤器

    用于检测并记录潜在的敏感信息泄露尝试，但不修改日志内容。
    可配合 SensitiveDataFilter 使用。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        检测日志中是否包含明显敏感字段名，并记录告警指标

        Returns:
            True（不阻断日志）
        """
        try:
            msg = record.getMessage()
            if isinstance(msg, str):
                lower_msg = msg.lower()
                # 检查是否直接包含敏感键名（作为独立单词出现）
                for key in LogConfig.SENSITIVE_KEYS:
                    if key in lower_msg:
                        LogMetrics.record("password_leak_attempts")
                        break
            return True
        except Exception:
            return True
