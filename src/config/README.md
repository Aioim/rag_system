# Config 模块 — RAG 企业级知识库问答配置管理

## 模块概述

Config 模块是 RAG 系统的统一配置中心，覆盖检索、分块、Embedding、LLM、会话、兜底策略等全部配置。

- **多源合并**：`{env}.yaml` → 环境变量 → CLI 覆盖，三级优先级
- **类型安全**：Pydantic v2 模型验证
- **热重载**：`settings.reload()` 运行时重载

## 文件结构

```
rag0709/
├── config/                  ← 配置数据（代码外）
│   ├── dev.yaml             # 开发环境配置（ENV=dev；其他环境对应 {env}.yaml）
│   ├── aliases.yaml         # 术语别名映射
│   └── prompts/             # Prompt 模板（按意图）
│       ├── concept.yaml
│       ├── procedure.yaml
│       ├── compare.yaml
│       └── lookup.yaml
│
└── src/config/              ← 加载代码
    ├── __init__.py          # 导出
    ├── settings.py          # 配置模型 + ConfigManager
    ├── yaml_loader.py       # YAML 多环境合并 + 缓存
    └── path.py              # PROJECT_ROOT
```

## 快速开始

```python
from config import settings

# 访问配置
settings.retrieval.top_k        # 5
settings.embedding.model         # BAAI/bge-large-zh-v1.5
settings.llm.default            # deepseek-v4-pro

# 点号路径
settings.get('retrieval.rrf_k')  # 60

# CLI 覆盖
settings.apply_overrides('retrieval.top_k=10,debug=true')

# 导出
print(settings.to_yaml())

# 热重载
settings.reload()
```

## 配置优先级

```
CLI 覆盖  >  环境变量  >  {env}.yaml  >  代码默认值
```

环境变量使用双下划线 `__` 表示嵌套：
```bash
RETRIEVAL__TOP_K=10
LLM__API_KEY=sk-xxx
MILVUS__HOST=192.168.1.100
```

## 配置项速查

| 配置块 | 关键字段 | 说明 |
|--------|---------|------|
| `retrieval` | top_k, rrf_k, mmr_lambda, relevance_threshold_* | 检索与精排 |
| `chunking` | chunk_size, overlap, strategy | 文档分块 |
| `ingestion` | parsers, parsed_doc_dir, mineru | 离线文档处理 + 解析器选择 |
| `session` | ttl_hours, db_path, topic_switch_threshold | 会话管理 |
| `embedding` | model, device, batch_size, dimension | Embedding |
| `llm` | default, lightweight, api_key_env, temperatures | LLM 路由 |
| `generation` | dedup_threshold, max_context_chars, fact_check_enabled | 生成控制 |
| `web_search` | enabled, provider, timeout_seconds | 联网兜底 |
| `faiss` | index_type, metric_type, nlist, index_dir | 向量数据库 |
| `model` | cache_dir, default_models, hf_endpoint | 模型下载管理 |
| `finetune` | training/lora/distillation 超参 | 模型微调 & 蒸馏 |
| `fallback` | max_retrieval_rounds, no_answer_message | 兜底策略 |
| `api` | host, port, cors_origins | API 服务 |
| `log` | log_level, log_file, max_bytes, backup_count | 日志 |

## 多环境配置

系统直接读取 `config/{ENV}.yaml`（默认 `ENV=dev`，对应 `config/dev.yaml`）。不存在时回退到 Pydantic 模型代码默认值。

创建环境配置文件（如 `config/prod.yaml`）：

```yaml
env: prod
milvus:
  host: milvus-prod.internal
llm:
  default: deepseek-v4-pro
log:
  log_level: WARNING
```

```bash
ENV=prod python -c "from config import settings; print(settings.env)"  # prod
```

> 创建新环境时，复制 `config/dev.yaml` 为 `config/{env}.yaml` 后按需修改即可。
