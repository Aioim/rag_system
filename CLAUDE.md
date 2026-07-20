# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
每进行到下一个子任务时，同步更新文档，保持文档与代码的一致性

## 项目目标

构建企业级 RAG 知识库问答系统中台，以 API 形式供业务系统调用。

- **场景**：企业内部知识库问答（1K~10K 文档，含表格和图片）
- **部署**：混合部署 — FAISS 本地（第一期），后续可迁移至 Milvus；LLM 走云端 API
- **核心能力**：多轮对话、查询改写、混合检索（向量+BM25）、Rerank 精排、Self-RAG 自适应检索、三级兜底（补充检索→联网搜索→诚实告知）、术语别名映射

## 框架版本要求

- `langchain` > 1.3.0
- `langgraph` >= 1.2.0

## 技术栈

| 组件 | 选型 |
|------|------|
| API 框架 | FastAPI + asyncio |
| 向量数据库 | FAISS (第一期)；后续可迁移至 Milvus 2.4+ |
| Embedding | BGE-large-zh-v1.5 (本地) |
| Reranker | BGE-Reranker v2-m3 Cross-Encoder (本地) |
| LLM | DeepSeek-v4-Pro / DeepSeek-v4-Flash (云端 API) |
| 会话存储 | SQLite |
| 配置管理 | Pydantic v2 + YAML + 环境变量三级合并 |
| 安全 | Fernet 内存加密 + .env 安全加载 + 日志脱敏 |
| 兜底处理 | DuckDuckGo / ddgs |

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
│   ├── dev.yaml               # 开发环境配置（ENV=dev 时读取；其他环境对应 {env}.yaml）
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
│   ├── model/                  # 模型下载管理与微调（详见 src/model/README.md）
│   │   ├── __init__.py          # 导出 + 全局单例 models
│   │   ├── downloader.py        # ModelDownloader — HuggingFace 下载引擎
│   │   ├── manager.py           # ModelManager — 模型管理器单例（含 finetune 方法）
│   │   └── finetune/            # 模型微调 & 蒸馏子模块
│   │       ├── __init__.py       # 子模块导出
│   │       ├── base.py           # BaseTrainer ABC + FinetuneResult/FinetuneInfo
│   │       ├── config.py         # Pydantic 微调配置（LoRA/Training/Distillation）
│   │       ├── data.py           # 数据加载 + JSONL 格式校验器
│   │       ├── aliases.py        # 模型类型别名映射
│   │       ├── embedding_trainer.py  # EmbeddingTrainer — 对比学习微调
│   │       ├── reranker_trainer.py   # RerankerTrainer — 分类微调
│   │       ├── llm_trainer.py        # LLMTrainer — SFT + 黑盒蒸馏
│   │       ├── cli.py               # CLI 命令注册
│   │       └── __main__.py          # CLI 入口（python -m model.finetune）
│   ├── logger/                # 安全日志系统（详见 src/logger/README.md）
│   │   ├── core.py            # 装饰器（@log_performance/@log_step）+ RequestLogger
│   │   ├── masking.py         # 正则脱敏引擎（密码/token/手机号/身份证等）
│   │   ├── lazy.py            # 线程安全延迟初始化 LazyLogger
│   │   └── filters.py         # SensitiveDataFilter + SecurityAuditFilter
│   ├── models/                # ✅ 共享数据模型（PipelineContext/Chunk/Session/API）
│   ├── session/               # ✅ SQLite 会话管理（详见 src/session/README.md）
│   ├── query/                 # ✅ 查询理解层（意图分类/上下文融合/查询改写）
│   │   ├── __init__.py          # 导出 + get_query_layer 单例工厂
│   │   ├── layer.py             # QueryUnderstandingLayer — Pipeline 主编排器
│   │   ├── intent_classifier.py # IntentClassifier — 意图分类 + 清晰度判断
│   │   ├── context_fuser.py     # ContextFuser — 多轮指代消解 + 追问补全
│   │   └── rewriters/           # 查询改写器（并行执行，合并去重）
│   │       ├── __init__.py       # QueryRewriter 编排器
│   │       ├── base.py           # BaseRewriter 基类（模板方法）
│   │       ├── hyde.py           # HyDERewriter — 生成假设答案
│   │       ├── keyword_rewriter.py # KeywordRewriter — 提取 BM25 关键词
│   │       └── synonym.py        # SynonymRewriter — 生成同义变体
│   ├── api/                   # ⬜ FastAPI 路由 + 中间件
│   ├── core/                  # ✅ RAG Pipeline 编排（查询理解→检索→兜底→生成→会话记录）
│   │   ├── __init__.py          # 导出 + get_rag_pipeline 单例工厂
│   │   ├── pipeline.py          # RAGPipeline — 全链路主编排器
│   │   └── fallback.py          # FallbackHandler — 薄包装层，委托至 src/fallback/
│   ├── retrieval/             # ✅ 混合检索（向量+BM25+RRF）+ Rerank + Self-RAG 自评
│   │   ├── __init__.py          # 导出 + get_retrieval_layer 单例工厂
│   │   ├── layer.py             # RetrievalLayer — 检索层主编排器
│   │   ├── vector_retriever.py  # VectorRetriever — FAISS 向量检索
│   │   ├── bm25_retriever.py    # BM25Retriever — jieba 分词 BM25 检索
│   │   ├── fusion.py            # RRF 融合 + 去重截断
│   │   ├── expander.py          # ContextExpander — prev/next 分块上下文扩展
│   │   ├── reranker.py          # CrossEncoderReranker — 精排 + MMR 多样性
│   │   ├── evaluator.py         # SelfRAGEvaluator — 检索质量自评
│   │   └── store.py             # FAISSStore + BM25Index — 索引读写
│   ├── generation/            # ✅ 生成层（Prompt 组装 / 模型路由 / 生成 / 事实核查 / 引用标注）
│   │   ├── __init__.py          # 导出 + get_generation_layer 单例工厂
│   │   ├── layer.py             # GenerationLayer — 生成层主编排器
│   │   ├── prompt_assembler.py  # PromptAssembler — 去重/上下文组装/拼接
│   │   ├── llm_router.py        # LLMRouter — 按意图路由模型+加载模板
│   │   ├── fact_checker.py      # FactChecker — 断言拆解+逐条核查(Lightweight)
│   │   └── citation_builder.py  # CitationBuilder — Chunk → Source引用列表
│   ├── ingestion/             # ✅ 离线文档处理 Pipeline（详见 src/ingestion/README.md）
│   │   ├── __init__.py          # 导出 + create_default_pipeline 工厂
│   │   ├── __main__.py          # CLI 演示入口 (python -m ingestion)
│   │   ├── pipeline.py          # IngestionPipeline — Stage 编排器
│   │   ├── stage.py             # Stage 协议（name/fatal/run）
│   │   ├── context.py           # PipelineContext / StageError 数据容器
│   │   ├── parser.py            # ParserStage — docling 解析 PDF/Word/Markdown
│   │   ├── chunker.py           # ChunkerStage — 语义/固定/层级 三种分块策略
│   │   ├── embedder.py          # EmbedderStage — 批量 embedding
│   │   └── indexer.py           # FAISSIndexWriter — 索引持久化
│   └── fallback/              # ✅ 三级兜底处理（补充检索/联网搜索/诚实告知）
├── docs/superpowers/specs/    # 设计文档
├── tests/                       # 测试目录（pytest + pytest-asyncio）
│   ├── conftest.py              # 共享 fixtures（LLM mock 等）
│   ├── unit/
│   │   ├── config/              # 配置模块测试
│   │   ├── core/                # RAG Pipeline + 兜底处理测试
│   │   ├── generation/          # 生成层测试（含 conftest.py）
│   │   ├── ingestion/           # 文档处理各阶段测试
│   │   ├── model/               # 模型微调测试
│   │   ├── models/              # 数据模型测试（enums/chunk/document/session/api/context）
│   │   ├── query/               # 查询理解层测试
│   │   │   └── rewriters/       # 改写器测试
│   │   ├── retrieval/           # 检索层测试（含 conftest.py）
│   │   └── session/             # 会话管理测试
```

## 当前开发阶段

第1期（基础 + 查询理解 + 文档处理）已完成：config / security / logger / model / models / session / query / ingestion。
第2期进行中：retrieval 已完成；generation 已完成；core 已完成；model/finetune 已完成；fallback 已完成；API 待实现。

设计文档参见 `docs/superpowers/specs/2026-07-09-rag-enterprise-qa-design.md`，优化策略全景参见 `RAG优化策略全景图.md`。

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

配置优先级：**CLI 覆盖 > 环境变量(`RETRIEVAL__TOP_K=10`) > `{env}.yaml` > 代码默认值**

环境变量注入采用**白名单过滤**：仅识别以配置段根名（如 `RETRIEVAL`）开头的双层嵌套变量（含 `__`）、顶层标量 `ENV`/`DEBUG`，或带 `RAG__` 逃生前缀的任意变量。系统环境变量（`PATH`/`TEMP`/`OS` 等）不会再误入配置。

### 安全模块

```python
from security import secrets, load_secure_dotenv

load_secure_dotenv()             # 加载 .env，自动解密 ENC[...]；支持 export KEY= 前缀和行尾注释
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

模型存储在 `PROJECT_ROOT/local_models/` 下，以 `{org}/{model_name}` 为目录结构。首次运行需在 `.env` 中设置 `HUGGINGFACE_TOKEN=hf_xxx` 以访问 BGE 系列模型。

**微调 & 蒸馏**：

```python
from model import models

# Embedding 微调（对比学习）
result = models.finetune("embedding", data_path="data/finetune/triplets.jsonl")
# Reranker 微调（分类）
result = models.finetune("reranker", data_path="data/finetune/rerank_data.jsonl")
# LLM SFT / 蒸馏（云端大模型 → 本地小模型 LoRA）
result = models.finetune("llm", data_path="data/finetune/instructions.jsonl",
                         teacher="deepseek-v4-pro", alpha=0.3)
# → result.adapter_path = Path  # LoRA 适配器保存路径

# 管理已微调适配器
models.list_finetuned()               # → {"my-lora": FinetuneInfo, ...}
models.get_finetuned_path("my-lora")  # → Path 或 None
models.remove_finetuned("my-lora")
```

CLI 入口：`python -m model.finetune <type> --data <path> [--name <n>] [--teacher <t>]`

训练数据格式（JSONL）：

| 模型类型 | 必需字段 |
|---------|----------|
| embedding | query, positive, negative |
| reranker | query, document, label (0/1) |
| llm | instruction, input, output |

微调配置由 `config/{env}.yaml` 的 `finetune:` 段控制（LoRA rank/训练轮次/学习率/蒸馏温度等），通过 `python -m model.finetune config` 查看。

### 查询理解模块

```python
from query import get_query_layer, reset_query_layer

# 初始化（首次调用时创建单例，temperature 参数控制各组件确定性）
layer = get_query_layer(llm, session_manager)

# 基础查询（无会话上下文）
ctx = await layer.process("什么是RAG？")
# → ctx.intent, ctx.query, ctx.rewritten_queries, ctx.needs_clarification

# 多轮对话（自动指代消解 + 追问补全）
ctx = await layer.process("需要什么材料？", session_id="s1")
# → ctx.query = "申请年假需要什么材料？"  (自动补全)
# → ctx.session  (自动附加)

# 特定知识库
ctx = await layer.process("配置手册", collection="tech_docs")

# 测试用重置
reset_query_layer()
```

**Pipeline 流程**：别名映射 → 意图分类+清晰度判断 → 多轮上下文融合 → 查询改写(并行)

**组件温度约定**：

| 组件 | temperature | 原因 |
|------|-------------|------|
| IntentClassifier | 0 | 意图分类需确定性 |
| ContextFuser | 0 | 指代消解需确定性 |
| KeywordRewriter | 0 | BM25 关键词需幂等 |
| HyDERewriter | 0.3 | 假设答案需受控创意 |
| SynonymRewriter | 0.3 | 同义变体需多样性 |

### 检索模块

```python
from retrieval import get_retrieval_layer, reset_retrieval_layer

layer = get_retrieval_layer()      # 单例；embedding/rerank 模型首次检索时懒加载
ctx = await layer.retrieve(ctx)    # 输入查询理解层产出的 PipelineContext
# → ctx.candidates      粗召回（向量+BM25 → RRF 融合去重截断）
# → ctx.reranked        CrossEncoder 精排 + MMR 后的最终 top_k
# → ctx.retrieval_eval  SUFFICIENT / NEED_MORE / INSUFFICIENT
reset_retrieval_layer()            # 测试用重置（同时清空 store 缓存）
```

**Pipeline 流程**：两路并行召回（每条改写 query 各跑向量+BM25，top_k×2）→ RRF 融合去重（截断至 `max_rerank_candidates`）→ prev/next 上下文扩展 → CrossEncoder 精排（对 `ctx.query`）+ MMR → Self-RAG 自评。

BM25 索引启动时从 docstore 内存构建（jieba 分词）；索引更新后调用 `store.reload()` 自动触发 BM25 重建。

### 生成模块

```python
from generation import get_generation_layer, reset_generation_layer

layer = get_generation_layer(llm)   # 单例；LLM 由上层 core 创建并注入
ctx = await layer.generate(ctx)     # 输入 retrieval 层产出的 ctx（含 reranked + intent）
# → ctx.answer        LLM 生成的回答（含事实核查警示标注）
# → ctx.sources       引用来源列表
# → ctx.confidence    置信度（0.6*rerank_avg + 0.4*fact_pass_rate）
# → ctx.assembled_prompt  组装后的完整 prompt（调试用）
reset_generation_layer()            # 测试用重置
```

**Pipeline 流程**：INSUFFICIENT 短路（不调 LLM）→ NEED_MORE 正常生成+partial 标注 → SUFFICIENT 完整流程（组装 → 路由 → 生成 → 核查 → 引用）。

**路由规则**（纯规则，本期）：lookup/procedure → lightweight（Flash）；concept/compare → default（Pro）。温度从 `settings.llm.temperatures` 按 intent 自动选取。

**事实核查**：调用轻量模型拆分答案为断言列表，逐条判断 supported/unsupported/contradicted，在答案末尾注入警示标注。核查失败不阻塞答案返回。

**配置**（`settings.generation`）：
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `dedup_threshold` | 0.85 | 上下文去重余弦阈值 |
| `max_context_chars` | 9000 | 上下文预算（≈6000 tokens） |
| `fact_check_enabled` | true | 事实核查开关 |

### Core 模块

```python
from core import get_rag_pipeline, reset_rag_pipeline

# 初始化（首次调用时创建单例，llm + session_manager 注入）
pipeline = get_rag_pipeline(llm, session_manager)

# 执行完整 RAG 问答链路
ctx = await pipeline.run("什么是RAG？")
# → ctx.answer         LLM 生成的回答（含事实核查标注）
# → ctx.sources        引用来源列表
# → ctx.confidence     置信度
# → ctx.is_fallback    是否触发兜底
# → ctx.fallback_level 兜底级别（none / partial / web_search / no_answer）

# 多轮对话
ctx = await pipeline.run("需要什么材料？", session_id="s1")

# 指定知识库
ctx = await pipeline.run("配置手册", collection="tech_docs")

# 模糊问题短路
ctx = await pipeline.run("帮帮我")
# → ctx.needs_clarification = True
# → ctx.clarification_question = "您想了解哪方面内容？"

reset_rag_pipeline()  # 测试用重置
```

**Pipeline 流程**：查询理解（别名映射→意图分类→上下文融合→查询改写）→ 检索（向量+BM25→RRF→Rerank→Self-RAG）→ 兜底处理（NEED_MORE→补充检索；INSUFFICIENT→联网搜索/诚实告知）→ 生成（组装→路由→生成→核查→引用）→ 会话记录。

**各层异常独立降级**：每层失败时记录日志并继续，不中断 Pipeline。

### Fallback 模块

```python
from fallback import get_fallback_handler, reset_fallback_handler

# 获取兜底处理器单例（含 WebSearcher + SupplementaryRetriever）
handler = get_fallback_handler()

# NEED_MORE → 补充检索（放宽 top_k 重新检索）
ctx = await handler.handle(ctx, retrieval_layer)
# → ctx.retrieval_eval 可能变为 SUFFICIENT 或仍为 NEED_MORE
# → ctx.fallback_level = FallbackLevel.PARTIAL

# INSUFFICIENT → 联网搜索 → 诚实告知
ctx = await handler.handle(ctx)
# → 搜索成功: ctx.fallback_level = FallbackLevel.WEB_SEARCH, ctx.answer = 搜索结果
# → 搜索失败: ctx.fallback_level = FallbackLevel.NO_ANSWER, ctx.answer = 兜底消息

# 单独使用搜索器
from fallback import WebSearcher
searcher = WebSearcher()
result = await searcher.search("Python RAG框架")
# → 拼接的搜索结果文本，失败返回 None

# 单独使用补充检索器
from fallback import SupplementaryRetriever
supp = SupplementaryRetriever()
ctx = await supp.retrieve(ctx, retrieval_layer)
# → 合并了放宽 top_k 后的新检索结果

reset_fallback_handler()  # 测试用重置
```

**兜底链路**：NEED_MORE（补充检索→标记 PARTIAL）→ INSUFFICIENT（联网搜索→WEB_SEARCH 或 NO_ANSWER）。

**配置**（`settings.web_search` / `settings.fallback`）：
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `web_search.enabled` | true | 联网搜索开关 |
| `web_search.provider` | duckduckgo | 搜索提供商 |
| `web_search.timeout_seconds` | 10 | 搜索超时 |
| `fallback.max_retrieval_rounds` | 2 | 补充检索最大轮次 |
| `fallback.no_answer_message` | "抱歉…" | 诚实告知消息 |

### Ingestion 模块

```python
from ingestion import create_default_pipeline

# 组装默认 pipeline（Parser → Chunker → Embedder → FAISSIndexWriter）
pipeline = create_default_pipeline()   # embedding 模型懒加载，跨调用缓存

# 处理文档并写入索引
ctx = await pipeline.run(Path("docs/员工手册.md"), collection="hr_docs")
# → ctx.document        原始文档信息
# → ctx.document.metadata["parsed_md_path"]  解析后 Markdown 文件路径
# → ctx.chunks          生成的分块（含 embedding）
# → ctx.status          "done" / "failed"
# → ctx.errors          非致命错误列表
# → ctx.metadata        各阶段耗时 (parser_ms / chunker_ms / embedder_ms)
```

**Pipeline 流程**：Parser（按 `ingestion.parsers` 配置选择后端：docling / pymupdf4llm / direct → Markdown 文本 → 写入 `parsed_doc_dir/{doc_id}.md`）→ Chunker（三种策略：语义/SentenceTransformer、固定窗口、Markdown 层级）→ Embedder（批量编码，写回 chunk.embedding）→ FAISSIndexWriter（向量+docstore 持久化；同 doc_id 重复写入时自动替换旧向量，避免孤儿向量堆积）

**Chunker 策略说明**：

| 策略 | 实现 | 适用场景 |
|------|------|----------|
| SemanticChunker | SemanticChunker | 通用文档，自适应语义边界 |
| FixedWindowChunker | 固定窗口 + 滑动步长 | 结构化较弱的文本 |
| HierarchicalChunker | 按 Markdown 标题层级 | 层级清晰的文档 |

**Stage 协议**：每个阶段实现 `name`（标识）、`fatal`（是否中断 pipeline）、`async def run(ctx) -> PipelineContext`。写入已嵌入的 chunk 时 EmbedderStage 自动跳过幂等。

**CLI 演示**：`python -m ingestion` 内置员工手册示例文档，自动检查/下载 embedding 模型，输出处理统计和索引文件。

**与 retrieval 的关系**：ingestion 负责将原始文档处理后写入 FAISS 索引（离线）；retrieval 负责从索引中检索（在线）。两者通过 FAISSStore 读写同一套 `{index_dir}/{collection}/` 目录结构。

**解析器后端**（`src/ingestion/parsers/`）：

| 解析器 | name | 支持格式 | 依赖 |
|--------|------|----------|------|
| `DoclingParser` | `docling` | pdf, docx, doc, pptx, ppt, html | docling>=2.0 |
| `PyMuPDF4LLMParser` | `pymupdf4llm` | pdf | pymupdf4llm>=0.2 |
| `MinerUParser` | `mineru` | pdf | magic-pdf>=0.6 |
| `DirectParser` | `direct` | md, markdown, txt | 无 |

切换 PDF 解析器：修改 `config/{env}.yaml` 中 `ingestion.parsers.pdf` 为 `pymupdf4llm` 或 `mineru`。

**配置**（`settings.ingestion`）：
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `parsed_doc_dir` | `data/parsed_docs` | 解析后 Markdown 输出目录 |
| `parsers` | `{pdf: docling, docx: docling, md: direct, ...}` | 文件扩展名 → 解析器名称映射 |
| `mineru.device` | `cpu` | MinerU 设备: cpu / cuda / mps |
| `mineru.models_dir` | `local_models/mineru` | MinerU 模型权重目录 |

### 会话模块

```python
from session import SessionManager

sm = SessionManager()
session = sm.get_or_create()           # 创建新会话
session = sm.get_or_create("abc-123")  # 获取已有会话
msg = sm.add_message("abc-123", "user", "什么是RAG？")
ctx = sm.get_context("abc-123")        # 获取对话上下文（含摘要）

# 归档历史消息（软删除，可溯源；非物理删除）
full = sm.store.get_messages("abc-123", include_archived=True)
```

**话题切换 / 上下文压缩**：旧消息标记 `archived=1`（软删除）而非物理删除，历史数据可通过 `get_messages(include_archived=True)` 溯源。会话读写通过 `asyncio.to_thread()` 移出事件循环线程，避免阻塞 asyncio。

### 多环境

创建 `config/prod.yaml` 覆盖默认值，通过 `ENV=prod` 环境变量切换。
