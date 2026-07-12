"""
模型管理模块 — 统一下载和管理 embedding / rerank / 本地 LLM 模型

使用示例：
    from model import models

    # 下载所有默认模型
    models.download_all()

    # 按类型下载
    models.download("embedding")

    # 查询
    models.status()              # → {"embedding": True, "rerank": False, "llm": False}
    models.get_path("embedding") # → Path 或 None
    models.list_downloaded()     # → {model_id: local_path, ...}
"""

__version__ = "1.0.0"

from .manager import ModelManager, models
from .downloader import ModelDownloader

__all__ = [
    # 核心入口
    "models",
    "ModelManager",
    "ModelDownloader",
    "__version__",
]
