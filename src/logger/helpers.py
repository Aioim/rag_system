"""工具函数模块"""

import inspect
from pathlib import Path
from typing import Tuple, Any

# ==================== 辅助函数：精准获取调用位置（跳过框架层） ====================
# 缓存：保存调用者信息
_caller_info_cache = {}


def _get_caller_info(skip_frames: int = 2) -> Tuple[str, str, int]:
    """
    获取调用者信息（智能跳过框架层）

    返回: (filename, func_name, lineno)
    """
    try:
        frame = inspect.currentframe()
        for _ in range(skip_frames):
            if frame and frame.f_back:
                frame = frame.f_back
            else:
                break

        if frame:
            code = frame.f_code
            # 生成缓存键：基于文件名和行号
            cache_key = (code.co_filename, frame.f_lineno)
            
            # 尝试从缓存获取
            if cache_key in _caller_info_cache:
                return _caller_info_cache[cache_key]
            
            filename = Path(code.co_filename).name

            # 跳过标准库/框架文件
            skip_patterns = [
                '/contextlib.py', '\\contextlib.py',
                '/decorator.py', '\\decorator.py',
                '/logging/', '\\logging\\',
                'secure_logger.py', 'password_guard.py'
            ]
            if any(pattern in code.co_filename for pattern in skip_patterns):
                # 继续向上查找业务代码
                while frame and frame.f_back:
                    frame = frame.f_back
                    code = frame.f_code
                    if not any(pattern in code.co_filename for pattern in skip_patterns):
                        filename = Path(code.co_filename).name
                        break

            result = (filename, code.co_name, frame.f_lineno)
            # 缓存结果
            if len(_caller_info_cache) < 1024:  # 限制缓存大小
                _caller_info_cache[cache_key] = result
            return result

        return "unknown.py", "unknown", 0

    except Exception:
        return "unknown.py", "unknown", 0


def _get_actual_module_name(frame_or_func: Any) -> str:
    """获取实际模块名(__main__ → 文件名）)"""
    try:
        if hasattr(frame_or_func, '__module__') and hasattr(frame_or_func, '__code__'):
            if frame_or_func.__module__ != "__main__":
                return frame_or_func.__module__
            try:
                file_path = Path(inspect.getfile(frame_or_func)).resolve()
                if not str(file_path).startswith('<'):
                    return file_path.stem
            except Exception:
                pass

        if hasattr(frame_or_func, 'f_code'):
            code = frame_or_func.f_code
            filename = code.co_filename
            if filename and not filename.startswith('<') and filename != __file__:
                return Path(filename).stem

        return "script"

    except Exception:
        return "script"
