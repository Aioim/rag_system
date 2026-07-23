"""
模型管理模块 — 统一下载和管理 embedding / rerank / 本地 LLM 模型，
以及微调和蒸馏训练。

使用示例：
    from model import models

    # 本地 LLM 推理
    output = models.generate("你好，请介绍一下自己")
    llm = models.local_llm          # 获取 LocalLLM 实例

    # 适配器：接入管线
    from model.llm_adapter import LocalLLMAdapter
    adapter = LocalLLMAdapter(models.local_llm)

    # 下载
    models.download_all()
    models.download("embedding")

    # 微调
    result = models.finetune("embedding", data_path="data/finetune/triplets.jsonl")

    # 蒸馏
    result = models.finetune("llm", data_path="...", teacher="deepseek-v4-pro")

    # 管理
    models.list_finetuned()
    models.get_finetuned_path("my-lora")
    models.remove_finetuned("my-lora")
"""

__version__ = "1.2.0"

from . import inference
from .downloader import (
    AutoStrategy,
    DownloadStrategy,
    HfStrategy,
    ModelDownloader,
    MsStrategy,
)
from .inference import LocalLLM, generate, get_local_llm
from .llm_adapter import LocalLLMAdapter
from .manager import ModelManager, models

__all__ = [
    "AutoStrategy",
    "DownloadStrategy",
    "HfStrategy",
    "LocalLLM",
    "LocalLLMAdapter",
    "ModelDownloader",
    "ModelManager",
    "MsStrategy",
    "__version__",
    "generate",
    "get_local_llm",
    "inference",
    "models",
]
