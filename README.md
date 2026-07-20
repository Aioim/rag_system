# RAG Service — 企业级知识库问答系统中台

基于 RAG（检索增强生成）的企业内部知识库问答系统，以 API 形式供业务系统调用。

## 技术栈

| 组件 | 选型 |
|------|------|
| API 框架 | FastAPI + asyncio |
| 向量数据库 | FAISS（第一期）；可迁移至 Milvus 2.4+ |
| Embedding | BGE-large-zh-v1.5（本地） |
| Reranker | BGE-Reranker v2-m3 Cross-Encoder（本地） |
| LLM | DeepSeek-v4-Pro / DeepSeek-v4-Flash（云端 API） |
| 文档解析 | Docling / PyMuPDF4LLM / MinerU（可插拔） |
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
│   ├── dev.yaml                # 开发环境配置（ENV=dev 时读取；其他环境对应 {env}.yaml）
│   ├── aliases.yaml            # 术语别名映射
│   └── prompts/                # 按意图分类的 Prompt 模板
├── src/
│   ├── config/                 # ✅ 配置加载模块
│   ├── security/               # ✅ 敏感信息管理（Fernet 加密）
│   ├── logger/                 # ✅ 安全日志系统
│   ├── model/                  # ✅ 模型下载与管理
│   ├── models/                 # ✅ 共享数据模型（PipelineContext/Chunk/Session 等）
│   ├── session/                # ✅ SQLite 会话管理
│   ├── query/                  # ✅ 查询理解层（意图分类/上下文融合/查询改写）
│   ├── api/                    # ⬜ FastAPI 路由 & 中间件
│   ├── core/                   # ✅ RAG Pipeline 编排
│   ├── retrieval/              # ✅ 混合检索 + Rerank
│   ├── generation/             # ✅ Prompt 组装 + LLM 生成
│   ├── ingestion/              # ✅ 离线文档处理（含 MinerU 解析）
│   └── fallback/               # ✅ 三级兜底
├── local_models/               # 模型文件（BGE 系列）
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

## 模块依赖关系

### 依赖层次图

上层依赖下层，同层模块通过 `PipelineContext` 传递数据。

```
                       ┌──────────┐
                       │    api   │  FastAPI 路由 + 中间件
                       └────┬─────┘
                            │
                       ┌────┴─────┐
                       │   core   │  Pipeline 编排（串联所有在线模块）
                       └────┬─────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
┌──────┴──────┐    ┌───────┴───────┐    ┌──────┴──────┐
│   session   │    │    query      │    │  fallback   │
│   会话管理   │◄───│  查询理解层    │    │  兜底处理    │
└─────────────┘    └───────┬───────┘    └──────┬──────┘
                           │                    ▲
                    ┌──────┴──────┐             │
                    │  retrieval  │─────────────┘
                    │   检索层     │  评估不足时触发
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │ generation  │
                    │   生成层     │
                    └─────────────┘

     ┌──────────────────────────────────────────────┐
     │  models  (PipelineContext / Chunk / Session)  │  ← 所有模块共享
     │  config  (YAML 配置 + 别名 + Prompt 模板)      │
     │  logger + security  (日志 + 脱敏 + 加密)       │
     └──────────────────────────────────────────────┘
```

### Pipeline 数据流

每个阶段消费上游填入 `PipelineContext` 的字段，并产出下游所需字段：

```
query ──► rewritten_queries, intent, session ──► retrieval
retrieval ──► candidates, reranked, retrieval_eval ──► generation
retrieval ──► retrieval_eval (INSUFFICIENT) ──► fallback ──► generation
generation ──► answer, sources, confidence ──► core ──► api
```

### 逐模块依赖

| 模块 | 状态 | 运行时注入/import 依赖 | 消费上游数据 |
|------|------|----------------------|-------------|
| **models** | ✅ | 无 | 无 — 纯数据结构 |
| **config** | ✅ | security（解密敏感配置） | 无 |
| **session** | ✅ | config, models.Session | 无 — 由 query/core 驱动写入 |
| **query** | ✅ | config.aliases, session.SessionManager, LLM | Pipeline 入口 |
| **retrieval** | ✅ | embedding/reranker 模型, FAISS | query → `rewritten_queries`, `intent`, `collection` |
| **generation** | ✅ | config/prompts/, LLM | retrieval → `reranked`, query → `intent` |
| **fallback** | ✅ | config.web_search, 联网搜索 API | retrieval → `retrieval_eval` |
| **core** | ✅ | query, retrieval, generation, fallback, session | 串联所有模块，传递 PipelineContext |
| **api** | ⬜ | core, models.api_models | HTTP Request → 路由到 core |
| **ingestion** | ✅ | embedding 模型, FAISS, MinerU (可选) | 离线链路，写入向量库供 retrieval 使用 |

### 实现顺序约束

```
第1期 ✅ models → config → session → query         （基础 + 查询理解）
第2期 ✅ retrieval → generation → fallback           （检索 + 生成 + 兜底）
第2期 ✅ core → ingestion → model/finetune           （编排 + 文档处理 + 微调）
第3期 ⬜ api                                          （对外接口）
```

关键约束：后续模块依赖 `PipelineContext` 中由上游填充的字段，必须按数据流顺序推进。

## 文档解析引擎

ingestion 模块支持可插拔解析器后端，通过 `config/{env}.yaml` 的 `ingestion.parsers` 按文件扩展名选择：

| 解析器 | name | 支持格式 | 依赖 |
|--------|------|----------|------|
| DoclingParser | `docling` | pdf, docx, doc, pptx, ppt, html | docling>=2.0 |
| PyMuPDF4LLMParser | `pymupdf4llm` | pdf | pymupdf4llm>=0.2 |
| **MinerUParser** | `mineru` | pdf | magic-pdf>=0.6 |
| DirectParser | `direct` | md, markdown, txt | 无 |

### MinerU 解析器（新增）

专为复杂排版 PDF（学术论文、技术手册、图文混排）优化：

```yaml
# config/{env}.yaml
ingestion:
  parsers:
    pdf: mineru  # 从 docling 切换为 mineru
  mineru:
    device: cpu           # cpu | cuda | mps
    models_dir: local_models/mineru
```

```python
from ingestion.parsers import get_parser

parser = get_parser("mineru")
markdown = parser.parse("document.pdf", output_dir="output/")
# 输出: output/document.md + output/document_images/*.jpg
```

**安装与模型下载**：

```bash
pip install magic-pdf[full-cpu] ultralytics doclayout-yolo rapidocr-onnxruntime rapid-table
# 模型下载（从 HuggingFace 镜像）
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from modelscope import snapshot_download
snapshot_download('opendatalab/PDF-Extract-Kit-1.0', cache_dir='local_models/mineru')
"
# 创建 magic-pdf.json 配置
```

首次运行时 MinerU 会自动下载 `hantian/layoutreader` 阅读顺序模型。

## 配置

优先级：**CLI 覆盖 > 环境变量(`RETRIEVAL__TOP_K=10`) > `{env}.yaml` > 代码默认值**

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
