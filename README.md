# RAG Service — 企业级知识库问答系统中台

基于 RAG（检索增强生成）的企业内部知识库问答系统，以 API 形式供业务系统调用。

## 技术栈

| 组件 | 选型 |
|------|------|
| API 框架 | FastAPI + asyncio |
| 向量数据库 | FAISS（第一期）；可迁移至 Milvus 2.4+ |
| Embedding | BGE-large-zh-v1.5（本地） |
| Reranker | BGE-Reranker v2-m3 Cross-Encoder（本地） |
| LLM | Claude Sonnet 5 / Haiku 4.5（云端 API） |
| 会话存储 | SQLite |
| 配置管理 | Pydantic v2 + YAML + 环境变量三级合并 |
| 安全 | Fernet 内存加密 + 日志脱敏 |

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url> && cd rag0709

# 2. 创建虚拟环境
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -e ".[retrieval]"    # 核心 + 检索增强
pip install -e ".[dev]"          # 开发依赖

# 4. 配置
cp config/.env.example config/.env
# 编辑 config/.env，填入 HUGGINGFACE_TOKEN 和 LLM_API_KEY

# 5. 下载模型
python -m model.downloader

# 6. 启动服务（待实现）
# uvicorn main:app --reload
```

## 项目结构

```
rag0709/
├── pyproject.toml              # 项目元数据 & 依赖
├── config/
│   ├── defaults.yaml           # 默认配置（所有可配项）
│   ├── aliases.yaml            # 术语别名映射
│   └── prompts/                # 按意图分类的 Prompt 模板
├── src/
│   ├── config/                 # 配置加载模块
│   ├── security/               # 敏感信息管理（Fernet 加密）
│   ├── logger/                 # 安全日志系统
│   ├── model/                  # 模型下载与管理
│   ├── api/                    # [待实现] FastAPI 路由 & 中间件
│   ├── core/                   # [待实现] RAG Pipeline 编排
│   ├── query/                  # [待实现] 查询理解层
│   ├── retrieval/              # [待实现] 混合检索 + Rerank
│   ├── generation/             # [待实现] Prompt 组装 + LLM 生成
│   ├── session/                # [待实现] SQLite 会话管理
│   ├── ingestion/              # [待实现] 离线文档处理
│   └── fallback/               # [待实现] 三级兜底
├── models/                     # 模型文件（BGE 系列）
├── data/                       # 运行时数据（SQLite 等）
├── logs/                       # 日志文件
└── docs/                       # 设计文档
```

## 架构

```
用户 Query → 安全护栏 → 查询理解层 → 检索层(多路召回+RRF+Rerank)
  → 上下文组装 → 生成层 → 事实核查 → 呈现层
                                    ↓ 资料不足
                              ┌── 补充检索
                              ├── 联网搜索
                              └── 诚实告知
```

在线 Pipeline（实时问答）与离线 Pipeline（文档处理）分离，离线走 ARQ 异步队列。

## 配置

优先级：**CLI 覆盖 > 环境变量(`RETRIEVAL__TOP_K=10`) > `{env}.yaml` > `defaults.yaml`**

```python
from config import settings

settings.retrieval.top_k          # 访问配置
settings.get("retrieval.rrf_k")   # 点号路径访问
settings.apply_overrides("retrieval.top_k=10")  # CLI 覆盖
```

多环境：创建 `config/prod.yaml`，通过 `ENV=prod` 环境变量切换。

## 开发

```bash
ruff check src/        # 代码检查
ruff format src/       # 代码格式化
pytest                 # 运行测试
```
