# Model 模块 — RAG 模型下载与管理

## 模块概述

Model 模块提供统一的模型下载和管理能力，支持从 HuggingFace Hub 下载三种模型：embedding（BGE）、reranker（BGE-Reranker）、本地轻量 LLM。

- **统一下载**：按类型名或 HuggingFace repo_id 下载模型
- **断点续传**：基于 huggingface_hub 内置的 resume_download
- **失败重试**：网络错误自动重试（指数退避：1s / 2s / 4s）
- **状态查询**：查看各模型类型的下载状态和本地路径

## 文件结构

```
rag0709/
├── config/
│   └── defaults.yaml          ← model 配置段（缓存目录/默认模型/重试参数）
└── src/model/
    ├── __init__.py            # 导出 + 全局单例 models
    ├── downloader.py          # ModelDownloader — HuggingFace 下载引擎
    ├── manager.py             # ModelManager — 模型管理器单例
    └── README.md
```

模型文件存储在 `PROJECT_ROOT/models/` 下，以 `{org}/{model_name}` 为目录结构，如 `models/BAAI/bge-large-zh-v1.5/`。

## 快速开始

```python
from model import models

# 查看缓存目录
print(models._cache_dir)

# 查看各类型的下载状态
print(models.status())
# → {"embedding": False, "rerank": False, "llm": False}

# 下载 embedding 模型（按类型名）
path = models.download("embedding")
print(path)  # → .../models/BAAI/bge-large-zh-v1.5

# 下载特定模型（按 HuggingFace repo_id）
path = models.download("BAAI/bge-reranker-v2-m3")

# 获取已下载模型的本地路径（不触发下载）
path = models.get_path("embedding")  # → Path 或 None

# 下载所有默认模型
models.download_all()

# 列出所有已下载模型
for model_id, local_path in models.list_downloaded().items():
    print(f"{model_id} → {local_path}")

# 删除模型
models.remove("rerank")
```

## 配置

模型下载行为由 `config/defaults.yaml` 中的 `model:` 段控制：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `cache_dir` | `models` | 模型下载根目录（相对于 PROJECT_ROOT） |
| `default_models.embedding` | `BAAI/bge-large-zh-v1.5` | 默认 embedding 模型 |
| `default_models.rerank` | `BAAI/bge-reranker-v2-m3` | 默认 reranker 模型 |
| `default_models.llm` | `Qwen/Qwen2.5-1.5B-Instruct` | 默认本地 LLM（预留） |
| `hf_token_env` | `HUGGINGFACE_TOKEN` | HF Token 环境变量名 |
| `max_retries` | `3` | 网络错误重试次数 |

在 `.env` 中设置 `HUGGINGFACE_TOKEN=hf_xxx` 可访问需要授权的模型（BGE 系列需要）。

## 与下游模块集成（后续开发）

embedder 和 reranker 模块在使用前检查模型是否已下载：

```python
from model import models

class Embedder:
    def __init__(self):
        path = models.get_path("embedding")
        if path is None:
            raise RuntimeError(
                "Embedding 模型未下载，请先运行: models.download('embedding')"
            )
        # self._model = SentenceTransformer(str(path))
```

## 依赖

```bash
pip install huggingface_hub
```
