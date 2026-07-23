"""
examples/_common.py — 示例共享工具

所有 example 文件的公共样板代码集中到此处。
使用方式（在各 example 文件顶部）:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from examples._common import PROJECT_ROOT, banner, check_embedding_model
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def banner(title: str) -> None:
    """打印分隔标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_embedding_model(auto_download: bool = False) -> bool:
    """检查 Embedding 模型是否已下载

    Args:
        auto_download: 是否自动下载（默认仅检查并提示）

    Returns:
        True 表示模型就绪可用
    """
    from model import models

    status = models.status()
    ready = status.get("embedding", False)

    if ready:
        print("  Embedding 模型: ✅ 已下载")
        return True

    print("  Embedding 模型: ⬜ 未下载")
    if auto_download:
        print("  正在下载（首次运行耗时较长，约 1-2 分钟）...")
        try:
            models.download("embedding")
            print("  ✅ 下载完成")
            return True
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            print("  请检查 HUGGINGFACE_TOKEN 环境变量或网络连接")
            return False
    else:
        print("  提示: 使用 models.download('embedding') 下载")
        return False
