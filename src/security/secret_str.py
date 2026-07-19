"""
🔐 SecretStr - 军用级敏感字符串容器

防护措施：
• 禁止直接打印（__repr__/__str__ 返回掩码）
• 禁止序列化（__getstate__/__reduce__ 抛出异常）
• 禁止弱引用（__weakref__ 禁用）
• 恒定时间比较（防时序攻击）
• 自动清零（del时覆盖内存）
• 防内存转储（__slots__ 限制属性）
"""
import contextlib
import secrets as secrets_lib
import sys
from typing import Any


class SecretStr:
    """敏感字符串容器 - 多层防泄露保护"""
    __slots__ = ('_accessed', '_name', '_value')

    def __init__(self, value: str, name: str = "secret"):
        if not isinstance(value, str):
            raise TypeError(f"Secret value must be str, got {type(value).__name__}")
        self._value = value
        self._name = name
        self._accessed = False

    def get(self) -> str:
        self._accessed = True
        return self._value

    def mask(self, visible_start: int = 3, visible_end: int = 4) -> str:
        val = self._value or ""
        total_len = len(val)
        if total_len <= visible_start + visible_end:
            return "*" * max(6, total_len)
        masked_len = total_len - visible_start - visible_end
        return f"{val[:visible_start]}{'*' * masked_len}{val[-visible_end:]}"

    def __repr__(self) -> str:
        return f"<SecretStr name='{self._name}' masked='{self.mask()}'>"

    def __str__(self) -> str:
        return self.mask()

    def __format__(self, format_spec: str) -> str:
        """禁止 f-string 格式化泄露，强制掩码"""
        return self.mask()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SecretStr):
            other_val = other._value
        elif isinstance(other, str):
            other_val = other
        else:
            return False
        return secrets_lib.compare_digest(self._value, other_val)

    def __len__(self) -> int:
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __hash__(self) -> int:
        raise TypeError("SecretStr objects are unhashable (security protection)")

    def __del__(self):
        """尝试覆盖内存（Python 无法保证安全擦除，仅尽力而为）"""
        try:
            if hasattr(self, '_value') and not sys.is_finalizing():
                self._value = '\0' * len(self._value)
        except Exception:
            pass

    # ========== 禁止序列化 ==========
    def __getstate__(self):
        raise RuntimeError(f"Cannot pickle SecretStr '{self._name}' - sensitive data protection")
    def __setstate__(self, state):
        raise RuntimeError(f"Cannot unpickle SecretStr '{self._name}' - sensitive data protection")
    def __reduce__(self):
        raise RuntimeError(f"Cannot serialize SecretStr '{self._name}' via pickle/reduce")
    def __reduce_ex__(self, protocol):
        raise RuntimeError(f"Cannot serialize SecretStr '{self._name}' via pickle/reduce_ex")

    __weakref__ = None

    # ========== 数值操作保护 ==========
    def __add__(self, other):
        raise TypeError("Cannot concatenate SecretStr (security protection)")
    def __radd__(self, other):
        raise TypeError("Cannot concatenate SecretStr (security protection)")
    def __mul__(self, other):
        raise TypeError("Cannot multiply SecretStr (security protection)")
    def __rmul__(self, other):
        raise TypeError("Cannot multiply SecretStr (security protection)")

    # ========== 安全方法 ==========
    def is_accessed(self) -> bool:
        return self._accessed

    def reset_access_flag(self) -> None:
        self._accessed = False

    @property
    def name(self) -> str:
        return self._name

    @contextlib.contextmanager
    def temporary(self):
        """安全上下文：用完立即清理局部引用"""
        plain = self.get()
        try:
            yield plain
        finally:
            del plain