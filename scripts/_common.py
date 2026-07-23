"""scripts/ 共享工具 — 项目根目录检测、控制台输出辅助"""
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    """从当前文件位置向上两级找到项目根目录 (scripts/ → rag0709/)"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _resolve_project_root()

# ---- 终端颜色 ----
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_CHECK = "✓"
_CROSS = "✗"
_WARN = "⚠"


def ok(msg: str) -> None:
    print(f"  {_GREEN}{_CHECK} {msg}{_RESET}")


def fail(msg: str) -> None:
    print(f"  {_RED}{_CROSS} {msg}{_RESET}")


def warn(msg: str) -> None:
    print(f"  {_YELLOW}{_WARN} {msg}{_RESET}")


def info(msg: str) -> None:
    print(f"  {msg}")


def banner(title: str) -> None:
    width = 50
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def check_python_version() -> tuple[bool, str]:
    """检查 Python 版本 >= 3.11"""
    major, minor = sys.version_info[:2]
    version_str = f"Python {major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 11):
        return True, version_str
    return False, version_str
