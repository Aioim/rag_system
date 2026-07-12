"""
模型管理模块 — 统一下载和管理 embedding / rerank / 本地 LLM 模型，
以及微调和蒸馏训练。

使用示例：
    from model import models

    # 下载
    models.download_all()
    models.download("embedding")

    # 微调
    result = models.finetune("embedding", data_path="data/finetune/triplets.jsonl")

    # 蒸馏
    result = models.finetune("llm", data_path="...", teacher="claude-sonnet-5")

    # 管理
    models.list_finetuned()
    models.get_finetuned_path("my-lora")
    models.remove_finetuned("my-lora")
"""

__version__ = "1.1.0"

from .manager import ModelManager, models
from .downloader import ModelDownloader

__all__ = [
    # 核心入口
    "models",
    "ModelManager",
    "ModelDownloader",
    "__version__",
]
