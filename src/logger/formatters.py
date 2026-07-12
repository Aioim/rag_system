"""
日志格式化器模块
提供安全格式化和敏感数据清理
"""
import logging
import re
import inspect
from pathlib import Path

from config import settings
LogConfig = settings.log


class SecurityFormatter(logging.Formatter):
    """统一日志格式：时间 级别 [文件:函数:行号] 消息"""
    _CRLF_PATTERN = re.compile(r'[\r\n\x1b\x9b]')
    _ANSI_ESCAPE = re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    STANDARD_FORMAT = "%(asctime)s %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    _main_module_cache = {}

    def format(self, record: logging.LogRecord) -> str:
        original_module = record.module
        original_filename = record.filename

        if LogConfig.replace_main_with_filename and record.module == "__main__":
            cache_key = (record.pathname, record.lineno)
            if cache_key in self._main_module_cache:
                record.module, record.filename = self._main_module_cache[cache_key]
            else:
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

        if isinstance(record.msg, str):
            record.msg = self._sanitize(record.msg)
        if record.args and isinstance(record.args, dict):
            record.args = {
                self._sanitize(k): self._sanitize(v) if isinstance(v, str) else v
                for k, v in record.args.items()
            }

        result = super().format(record)
        record.module = original_module
        record.filename = original_filename
        return result

    @staticmethod
    def _sanitize(text: str) -> str:
        text = SecurityFormatter._ANSI_ESCAPE.sub('', text)
        return SecurityFormatter._CRLF_PATTERN.sub(' ', text)

