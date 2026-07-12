# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

构建企业级 RAG 知识库问答系统中台，以 API 形式供业务系统调用。

- **场景**：企业内部知识库问答（1K~10K 文档，含表格和图片）
- **部署**：混合部署 — FAISS 本地（第一期），后续可迁移至 Milvus；LLM 走云端 API
- **核心能力**：多轮对话、查询改写、混合检索（向量+BM25）、Rerank 精排、Self-RAG 自适应检索、三级兜底（补充检索→联网搜索→诚实告知）、术语别名映射

## 框架版本要求

- `langchain` >= 1.4.0
- `langgraph` >= 1.2.0

## 技术栈

| 组件 | 选型 |
|------|------|
| API 框架 | FastAPI + asyncio |
| 向量数据库 | FAISS (第一期)；后续可迁移至 Milvus 2.4+ |
| Embedding | BGE-large-zh-v1.5 (本地) |
| Reranker | BGE-Reranker v2-m3 Cross-Encoder (本地) |
| LLM | Claude Sonnet 5 / Haiku 4.5 (云端 API) |
| 会话存储 | SQLite |
| 配置管理 | Pydantic v2 + YAML + 环境变量三级合并 |
| 安全 | Fernet 内存加密 + .env 安全加载 + 日志脱敏 |

## 架构

```
用户 Query → [安全护栏] → [查询理解层] → [检索层(多路召回+RRF+Rerank)]
  → [上下文组装] → [生成层] → [事实核查] → [呈现层]
                                              ↓ 资料不足
                                        [兜底: 联网搜索/告知]
```

在线 Pipeline（实时问答）和离线 Pipeline（文档解析→分块→Embedding→入 FAISS 索引）分离，离线走 Celery/ARQ 异步队列。

## 项目结构

```
rag0709/
├── config/                    # YAML 配置数据
│   ├── defaults.yaml          # 默认配置（检索/分块/LLM/FAISS/兜底等全部配置项）
│   ├── aliases.yaml           # 用户术语 → 标准术语映射（如 "工资条"→"薪资明细"）
│   └── prompts/               # 按意图分类的 Prompt 模板（concept/procedure/compare/lookup）
├── src/
│   ├── config/                # 配置加载模块（详见 src/config/README.md）
│   │   ├── settings.py        # Pydantic 配置模型 + ConfigManager 单例
│   │   ├── yaml_loader.py     # YAML 多环境合并 + LRU 缓存
│   │   ├── aliases.py         # 别名映射器（支持热重载）
│   │   └── path.py            # PROJECT_ROOT
│   ├── security/              # 敏感信息管理模块（详见 src/security/README.md）
│   │   ├── secrets_manager.py # Fernet 内存加密存储单例
│   │   ├── secret_str.py      # 防泄露字符串容器（禁止打印/序列化）
│   │   ├── secure_env_loader.py # 安全 .env 加载（自动解密 ENC[...]）
│   │   └── env_encryptor.py   # CLI 加密工具 + 编程接口
│   ├── model/                  # 模型下载与管理（详见 src/model/README.md）
│   │   ├── __init__.py          # 导出 + 全局单例 models
│   │   ├── downloader.py        # ModelDownloader — HuggingFace 下载引擎
│   │   └── manager.py           # ModelManager — 模型管理器单例
│   ├── logger/                # 安全日志系统（详见 src/logger/README.md）
│   │   ├── core.py            # 装饰器（@log_performance/@log_step）+ RequestLogger
│   │   ├── masking.py         # 正则脱敏引擎（密码/token/手机号/身份证等）
│   │   ├── lazy.py            # 线程安全延迟初始化 LazyLogger
│   │   └── filters.py         # SensitiveDataFilter + SecurityAuditFilter
│   ├── api/                   # [待实现] FastAPI 路由 + 中间件
│   ├── core/                  # [待实现] RAG Pipeline 编排
│   ├── query/                 # [待实现] 查询理解层（意图分类/改写/别名映射/上下文融合）
│   ├── retrieval/             # [待实现] 混合检索 + Rerank + 检索评估
│   ├── generation/            # [待实现] Prompt 组装 + LLM 路由 + 生成 + 事实核查
│   ├── session/               # [待实现] SQLite 会话管理
│   ├── ingestion/             # [待实现] 离线文档处理 Pipeline
│   ├── fallback/              # [待实现] 三级兜底处理
│   └── models/                # [待实现] 数据模型（Document/Chunk/Session/API）
└── docs/superpowers/specs/    # 设计文档
```

## 当前开发阶段

基础模块（config / security / logger / model）已完成，核心 RAG Pipeline 待实现。设计文档参见 `docs/superpowers/specs/2026-07-09-rag-enterprise-qa-design.md`，优化策略全景参见 `RAG优化策略全景图.md`。

## 开发要点

### 配置系统

```python
from config import settings, resolve_alias

settings.retrieval.top_k          # 访问配置
settings.get("retrieval.rrf_k")   # 点号路径访问
settings.apply_overrides("retrieval.top_k=10")  # CLI 覆盖
settings.reload()                 # 热重载
resolve_alias("工资条")           # → "薪资明细"
```

配置优先级：**CLI 覆盖 > 环境变量(`RETRIEVAL__TOP_K=10`) > `{env}.yaml` > `defaults.yaml` > 代码默认值**

### 安全模块

```python
from security import secrets, load_secure_dotenv

load_secure_dotenv()             # 加载 .env，自动解密 ENC[...]
secrets.set_secret("key", "v")   # 内存加密存储
secrets.get_secret("key")        # 动态解密
```

### 日志模块

```python
from logger import logger, security_logger, log_performance, log_step

@log_performance(threshold_ms=100)
def retrieve(query): ...
```

### 模型模块

```python
from model import models

models.status()                   # → {"embedding": False, "rerank": False, "llm": False}
models.download("embedding")      # 按类型下载默认模型
models.get_path("embedding")      # → Path 或 None（不触发下载）
models.list_downloaded()          # → {model_id: local_path, ...}
```

模型存储在 `PROJECT_ROOT/models/` 下，以 `{org}/{model_name}` 为目录结构。首次运行需在 `.env` 中设置 `HUGGINGFACE_TOKEN=hf_xxx` 以访问 BGE 系列模型。

### 多环境

创建 `config/prod.yaml` 覆盖默认值，通过 `ENV=prod` 环境变量切换。
