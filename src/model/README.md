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
│   └── {env}.yaml             ← model 配置段（缓存目录/默认模型/重试参数）
└── src/model/
    ├── __init__.py            # 导出 + 全局单例 models
    ├── downloader.py          # 下载策略（HfStrategy / MsStrategy / AutoStrategy）+ ModelDownloader
    ├── manager.py             # ModelManager — 模型管理器单例
    └── README.md
```

模型文件存储在 `PROJECT_ROOT/local_models/` 下，以 `{org}/{model_name}` 为目录结构，如 `local_models/BAAI/bge-large-zh-v1.5/`。

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
print(path)  # → .../local_models/BAAI/bge-large-zh-v1.5

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

模型下载行为由 `config/{env}.yaml` 中的 `model:` 段控制：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `cache_dir` | `local_models` | 模型下载根目录（相对于 PROJECT_ROOT） |
| `default_models.embedding` | `BAAI/bge-large-zh-v1.5` | 默认 embedding 模型 |
| `default_models.rerank` | `BAAI/bge-reranker-v2-m3` | 默认 reranker 模型 |
| `default_models.llm` | `Qwen/Qwen2.5-1.5B-Instruct` | 默认本地 LLM（预留） |
| `hf_token_env` | `HUGGINGFACE_TOKEN` | HF Token 环境变量名 |
| `max_retries` | `3` | 网络错误重试次数 |
| `download_source` | `auto` | 下载源: `huggingface` / `modelscope` / `auto` |

在 `.env` 中设置 `HUGGINGFACE_TOKEN=hf_xxx` 可访问需要授权的模型（BGE 系列需要）。

## 下载源切换

通过 `model.download_source` 配置选择下载源：

- **`huggingface`** — 使用 HuggingFace Hub（通过 `hf_endpoint` 可指定镜像站）
- **`modelscope`** — 使用 ModelScope（魔搭），国内直连免 token
- **`auto`**（默认）— 优先尝试 ModelScope，失败自动回退 HuggingFace

```python
# 编程方式切换（需在 ModelManager 初始化前设置）
from config import settings
settings.apply_overrides("model.download_source=modelscope")
```


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

## 模型微调 & 蒸馏

### 快速开始

```python
from model import models

# Embedding 微调
result = models.finetune("embedding", data_path="data/finetune/triplets.jsonl")

# Reranker 微调
result = models.finetune("reranker", data_path="data/finetune/rerank_data.jsonl")

# LLM SFT 微调
result = models.finetune("llm", data_path="data/finetune/instructions.jsonl")

# LLM 蒸馏（云端大模型 → 本地小模型）
result = models.finetune("llm", data_path="data/finetune/instructions.jsonl",
                         teacher="deepseek-v4-pro", alpha=0.3)
```

### CLI

```bash
# 微调 embedding
python -m model.finetune embedding --data data/finetune/triplets.jsonl --name my-emb

# 蒸馏 LLM
python -m model.finetune llm --data data/finetune/instructions.jsonl \
    --teacher deepseek-v4-pro --alpha 0.3

# 管理
python -m model.finetune list
python -m model.finetune info --name my-emb
python -m model.finetune remove --name my-emb
```

### 训练数据格式

| 模型类型 | JSONL 字段 | 示例 |
|---------|-----------|------|
| embedding | query, positive, negative | `{"query": "...", "positive": "...", "negative": "..."}` |
| reranker | query, document, label | `{"query": "...", "document": "...", "label": 1}` |
| llm | instruction, input, output | `{"instruction": "...", "input": "...", "output": "..."}` |

数据文件放在 `data/finetune/` 目录下。

### 蒸馏流程

1. 准备指令数据（instruction + input + output）
2. 用 `--teacher` 指定云端大模型，自动调用 API 生成教师答案
3. 混合教师答案和人工标注训练学生模型
4. 输出 LoRA 适配器到 `local_models/finetuned/`

### 配置

微调参数通过 `config/{env}.yaml` 的 `finetune:` 段控制，CLI 参数可覆盖。

### 依赖

```bash
pip install huggingface_hub
pip install rag-service[finetune]
```
