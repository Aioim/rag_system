"""
日志格式化器模块
提供安全格式化和敏感数据清理
"""
import inspect
import logging
import re
from pathlib import Path

from config import settings

from .masking import mask_sensitive_data


class SecurityFormatter(logging.Formatter):
    """统一日志格式：时间 级别 [文件:函数:行号] 消息"""
    _CRLF_PATTERN = re.compile(r'[\r\n\x1b\x9b]')
    _ANSI_ESCAPE = re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    STANDARD_FORMAT = "%(asctime)s %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    _main_module_cache: dict = {}  # noqa: RUF012

    def format(self, record: logging.LogRecord) -> str:
        original_module = record.module
        original_filename = record.filename
        original_msg = record.msg
        original_args = record.args

        try:
            if settings.log.replace_main_with_filename and record.module == "__main__":
                cache_key = (record.pathname, record.lineno)
                if cache_key in self._main_module_cache:
                    record.module, record.filename = self._main_module_cache[cache_key]
                else:
                    frame = None
                    try:
                        frame = inspect.currentframe()
                        depth = 0
                        while frame and depth < 10:
                            code = frame.f_code
                            filename = code.co_filename
                            if (filename and not filename.startswith('<') and
                                filename != __file__ and 'logging' not in filename):
                                module_name = Path(filename).stem
                                file_name = Path(filename).name
                                record.module = module_name
                                record.filename = file_name
                                self._main_module_cache[cache_key] = (module_name, file_name)
                                break
                            frame = frame.f_back
                            depth += 1
                    except Exception:
                        record.module = "script"
                        record.filename = "unknown.py"
                        self._main_module_cache[cache_key] = ("script", "unknown.py")
                    finally:
                        del frame

            if isinstance(record.msg, str):
                record.msg = self._sanitize(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        self._sanitize(k): self._sanitize(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, (list, tuple)):
                    record.args = tuple(
                        self._sanitize(v) if isinstance(v, str) else v
                        for v in record.args
                    )

            # 对格式化后的完整字符串脱敏：覆盖 msg 与 args 拼接后才出现的
            # 敏感形态（如 "token=%s" % tok），且不依赖改写格式串本身
            return mask_sensitive_data(super().format(record))
        finally:
            # 恢复现场：LogRecord 被所有 Handler 共享，
            # 永久变异会污染下游 Formatter（审查 C1/H6）
            record.module = original_module
            record.filename = original_filename
            record.msg = original_msg
            record.args = original_args

    @staticmethod
    def _sanitize(text: str) -> str:
        text = SecurityFormatter._ANSI_ESCAPE.sub('', text)
        return SecurityFormatter._CRLF_PATTERN.sub(' ', text)

