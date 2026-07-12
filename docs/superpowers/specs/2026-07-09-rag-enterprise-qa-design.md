# 企业级 RAG 知识库问答系统 — 设计文档

> 日期：2026-07-09
> 状态：设计完成，待评审

---

## 1. 需求概述

| 维度 | 决策 |
|------|------|
| 场景 | 企业内部知识库问答 |
| 文档规模 | 1K ~ 10K 文档，含表格和图片 |
| 部署方式 | 混合部署（Milvus 本地，LLM 走云端 API） |
| 交互形态 | API 中台（供业务系统调用） |
| 多轮对话 | 完整会话管理（上下文压缩、话题检测、指代消解） |
| 权限 | 无权限区分 |
| 技术栈 | Python（FastAPI + Milvus + BGE） |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      调用方（业务系统）                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/SSE
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     API 网关层 (FastAPI)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ 问答接口  │  │ 检索接口  │  │ 管理接口  │  │ 健康检查/监控 │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG 核心 Pipeline                        │
│                                                              │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐ │
│  │ 查询理解 │──▶│ 多路检索  │──▶│ Rerank   │──▶│ 上下文组装 │ │
│  │ 改写层   │   │ 召回层    │   │ 精排层    │   │ 生成层     │ │
│  └─────────┘   └──────────┘   └──────────┘   └───────────┘ │
│       │              │                                │      │
│       ▼              ▼                                ▼      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    会话管理 (SQLite)                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Milvus     │  │   LLM API    │  │  Web Search  │
│  向量库(本地) │  │  (云端/本地)  │  │  (兜底)      │
└──────────────┘  └──────────────┘  └──────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   离线 Pipeline（异步）                        │
│                                                              │
│  文档上传 → 解析清洗 → 多模态处理 → 分块 → Contextual        │
│  Retrieval → Embedding → 索引写入(Milvus)                     │
└─────────────────────────────────────────────────────────────┘
```

### 设计原则

- **在线/离线分离**：文档处理走异步队列，问答走实时链路，互不阻塞
- **每层可替换**：检索层可换向量库，生成层可换 LLM，pipeline 核心只做编排
- **混合部署**：Milvus 本地部署，LLM 默认走云端（可切换到本地模型）

---

## 3. API 设计

### 3.1 接口列表

```
POST   /api/v1/chat              # 问答（支持多轮）
POST   /api/v1/search            # 纯检索（不生成答案）
POST   /api/v1/documents/upload  # 文档上传（异步）
GET    /api/v1/documents/status/{task_id}  # 文档处理状态查询
DELETE /api/v1/sessions/{id}     # 清除会话
GET    /api/v1/health            # 健康检查
```

### 3.2 核心请求/响应

```python
class ChatRequest:
    query: str                    # 用户问题
    session_id: str | None        # 会话ID，为空则新建
    collection: str = "default"   # 知识库集合
    stream: bool = False          # 是否流式返回
    top_k: int = 5

class ChatResponse:
    answer: str
    sources: list[Source]
    session_id: str
    confidence: float             # 0~1
    is_fallback: bool             # 是否触发联网兜底

class Source:
    doc_id: str
    doc_title: str
    chunk_text: str               # 引用原文片段
    score: float

class SearchRequest:
    query: str
    collection: str = "default"
    top_k: int = 10

class SearchResponse:
    results: list[Source]
    search_type: str              # "hybrid" | "semantic" | "keyword"
```

---

## 4. 核心数据模型

### 4.1 PipelineContext

```python
class PipelineContext:
    query: str                          # 原始查询
    rewritten_queries: list[str]        # 改写后的多路查询
    intent: Intent                      # 意图分类结果
    collection: str                     # 目标知识库
    candidates: list[Chunk]             # 粗召回结果
    reranked: list[Chunk]               # 精排后结果
    session: Session | None
    assembled_prompt: str
    answer: str
    sources: list[Source]
    confidence: float
    retrieval_eval: RetrievalEval       # SUFFICIENT / NEED_MORE / INSUFFICIENT
    fallback_level: str                 # "" / "partial" / "web_search" / "no_answer"
    is_fallback: bool
    needs_clarification: bool
    clarification_question: str | None
    metadata: dict                      # 耗时、token 消耗等
```

### 4.2 Chunk

```python
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    prev_chunk_id: str | None       # 上一 chunk，支持向左扩展上下文
    next_chunk_id: str | None       # 下一 chunk，支持向右扩展上下文
    context_summary: str | None     # 所属文档的上下文摘要 (Contextual Retrieval)
    embedding: list[float] | None
    rerank_score: float = 0.0
    metadata: dict                  # 页码、章节标题等
```

### 4.3 Session

```python
class Session:
    session_id: str
    messages: list[Message]         # 最近 N 轮对话
    context_summary: str | None     # 对话摘要（长对话压缩用）
    current_topic: str | None       # 当前话题
    topic_embedding: list[float]    # 话题 embedding（切换检测用）
    created_at: datetime
    last_active: datetime
```

### 4.4 Document

```python
class Document:
    doc_id: str
    title: str
    file_type: str                  # pdf / docx / md / html
    collection: str                 # 所属知识库集合
    raw_path: str
    status: str                     # pending / parsing / multimodal / chunking
                                    #   / contextual / embedding / done / failed
    metadata: dict
    created_at: datetime
```

---

## 5. Pipeline 详细设计

### 5.1 查询理解层

```
用户 Query
    │
    ├── 1. 别名映射 → "工资条" → "薪资明细"
    ├── 2. 意图分类 → 概念理解/操作步骤/对比分析/精确查找
    ├── 3. 清晰度判断 → 模糊则短路，返回澄清问题
    ├── 4. 多轮上下文融合（有 session 时）
    │      ├── 指代消解
    │      ├── 追问补全
    │      └── 话题切换检测
    └── 5. 查询改写（并行）
           ├── HyDE 假设答案
           ├── 关键词提取（给 BM25）
           └── 同义词扩展
                  │
                  ▼
           多路 query → 检索层
```

**意图分类 Prompt 模板**（4 种，按需扩展）：

| 意图 | 角色设定 | 格式要求 | 温度 |
|------|---------|---------|------|
| 概念理解 | 擅长用简洁语言解释概念 | 定义→要点→案例 | 0.3 |
| 操作步骤 | 擅长给出准确操作指引 | 按步骤编号，标注注意事项 | 0.0 |
| 对比分析 | 擅长多维度对比 | 表格对比→选型建议 | 0.2 |
| 精确查找 | 擅长查找具体数据/条款 | 直接给出数据和出处 | 0.0 |

### 5.2 检索层

```
rewritten_queries
    │
    ├── 路由：根据 intent + 配置选择 collection
    │     default / tech / policy / faq
    │
    ├── 三路并行召回（每条 query 都跑）
    │   ├── 向量检索 (Milvus dense)  → top_k × 2
    │   ├── BM25 检索 (Milvus sparse) → top_k × 2
    │   └── 摘要索引检索              → top_k × 1
    │
    ├── RRF 融合去重 → score = Σ 1/(k + rank_i)
    │
    └── 上下文扩展 → 通过 prev_chunk_id/next_chunk_id
        向前后各拉 expansion_window 个 chunk（默认 1，可配置）
              │
              ▼
         candidates → Rerank 层
```

### 5.3 Rerank 层

```
candidates
    │
    ├── Cross-Encoder 精排 (BGE-Reranker, 本地)
    │
    ├── MMR 多样性过滤 (mmr_lambda 默认 0.7, 可配置)
    │
    └── 截断 → 最终 top_k 条 → 上下文组装层
```

### 5.4 检索自评 (Self-RAG)

```python
class RetrievalEval(Enum):
    SUFFICIENT  = "sufficient"   # avg_score >= 0.5, 直接生成
    NEED_MORE   = "need_more"    # 0.3 <= avg_score < 0.5, 补充检索
    INSUFFICIENT = "insufficient" # avg_score < 0.3, 触发兜底
```

### 5.5 上下文组装层

```
reranked chunks
    │
    ├── 去重：余弦相似度 > 0.85 → 保留高分
    ├── Token 预算分配 (max_tokens = 6000)
    ├── Lost in the Middle 缓解：Top-1 文档全文置顶
    └── 冲突检测：多来源矛盾标注
              │
              ▼
         assembled prompt → 生成层
```

### 5.6 生成层

- **模型路由**：规则 + 小模型自评 → 简单问题用轻量模型，复杂问题用大模型
- **温度控制**：精确查找 0.0，概念理解 0.3
- **输出格式**：根据意图类型自动选择结构化模板

### 5.7 事实核查层（一期）

```
生成的 answer
    │
    ├── 拆解为独立断言列表
    ├── 逐条核查是否被检索资料支撑
    │    ├── supported    → 通过
    │    ├── unsupported  → 置信度降权 + 标注
    │    └── contradicted → 标注冲突
    └── 注入警示标注 → 最终 answer
```

---

## 6. 兜底策略（三级兜底链）

```
检索评估结果
    │
    ├── SUFFICIENT (avg_score >= 0.5)
    │   → 直接生成 + 事实核查
    │
    ├── NEED_MORE (0.3 <= avg_score < 0.5)
    │   → 放宽 top_k 补充检索一轮
    │   → 仍不足 → 降级标注 (partial)
    │
    └── INSUFFICIENT (avg_score < 0.3)
        → 联网搜索回退
            ├── 成功 → 融合本地+网络结果生成 (标注 is_fallback=true)
            └── 失败 → 诚实告知 + 建议补充文档
```

注：阈值 0.3/0.5 为默认值，可通过配置文件调整。

---

## 7. 会话管理

- **存储**：SQLite（本地轻量，无额外依赖）
- **TTL**：默认 2 小时（可配置）
- **上下文压缩**：超过 token 上限时，用轻量模型将早期消息转为摘要
- **话题切换检测**：新消息 embedding 与当前话题 embedding 相似度 < 阈值（默认 0.5）→ 重置摘要 + 只保留最近 2 轮对话
- **持久化**：每次对话轮次后写入 SQLite，服务重启不丢失

---

## 8. 离线文档处理 Pipeline

```
文档上传 → 解析 → 多模态处理 → 语义分块
    → Contextual Retrieval → Embedding → 索引写入
```

### 各阶段说明

| 阶段 | 说明 |
|------|------|
| 解析 | PDF(PyMuPDF) / Word(python-docx) / Markdown / HTML → 统一中间格式 |
| 多模态 | 表格→Markdown；图片→多模态模型描述+OCR |
| 分块 | 按标题层级+语义边界，512 tokens，重叠 64 tokens |
| Contextual Retrieval | 每个 chunk 拼接所属文档的上下文摘要后再 embedding |
| Embedding | BGE-large-zh-v1.5 本地，批量处理 |
| 索引写入 | Milvus dense + sparse 双索引 + 摘要索引 |

### 异步处理

- 基于 Celery/ARQ 异步任务队列
- 状态可查询：pending → parsing → multimodal → chunking → contextual → embedding → done/failed
- 失败自动记录错误日志

---

## 9. 别名映射

- 启动时从 `config/aliases.yaml` 加载
- 支持热更新（文件变更自动 reload）
- 格式：`用户术语 → 标准术语`，如 `"工资条" → "薪资明细"`

---

## 10. 配置结构

```yaml
# 检索
retrieval:
  top_k: 5
  expansion_window: 1          # 上下文扩展窗口（可配置）
  rrf_k: 60
  mmr_lambda: 0.7
  relevance_threshold:         # 兜底阈值（可调）
    sufficient: 0.5
    need_more: 0.3

# 分块
chunking:
  chunk_size: 512
  overlap: 64
  strategy: "semantic"

# 会话
session:
  ttl_hours: 2
  max_history_rounds: 10
  max_context_tokens: 4000
  db_path: "data/sessions.db"

# Embedding
embedding:
  model: "BAAI/bge-large-zh-v1.5"
  device: "cuda"
  batch_size: 32

# LLM 路由
llm:
  default: "claude-sonnet-5"
  lightweight: "claude-haiku-4-5"
  local: null

# 联网搜索
web_search:
  enabled: true
  provider: "duckduckgo"
  timeout_seconds: 10
```

---

## 11. 工程结构

```
rag-service/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── config/
│   ├── default.yaml
│   ├── aliases.yaml
│   └── prompts/
│       ├── concept.yaml
│       ├── procedure.yaml
│       ├── compare.yaml
│       └── lookup.yaml
├── src/
│   ├── main.py
│   ├── api/routes/           # chat, search, documents, sessions
│   ├── api/middleware/        # logging, error_handler
│   ├── core/                 # pipeline.py, context.py
│   ├── query/                # alias_mapper, intent_classifier, clarity_checker,
│   │   │                    # context_fuser, rewriters/
│   ├── retrieval/            # hybrid_retriever, milvus_client, bm25_handler,
│   │   │                    # reranker, context_expander, retrieval_evaluator
│   ├── generation/           # prompt_assembler, llm_router, generator,
│   │   │                    # fact_checker, citation_builder
│   ├── session/              # manager.py, store.py
│   ├── ingestion/            # pipeline.py, parsers/, multimodal.py,
│   │   │                    # chunker.py, contextual_retrieval.py, embedder.py
│   ├── fallback/             # handler.py, web_search.py
│   └── models/               # document.py, chunk.py, session.py, api.py
└── tests/
    ├── unit/
    └── integration/
```

---

## 12. 技术选型汇总

| 组件 | 选型 | 部署位置 |
|------|------|---------|
| API 框架 | FastAPI + asyncio | 本地 |
| 向量数据库 | Milvus 2.4+ (dense + sparse) | 本地 |
| Embedding | BGE-large-zh-v1.5 | 本地 |
| Reranker | BGE-Reranker v2-m3 (Cross-Encoder) | 本地 |
| LLM (复杂) | Claude Sonnet 5 / GPT-4o | 云端 API |
| LLM (轻量) | Claude Haiku 4.5 / GPT-4o-mini | 云端 API |
| 会话存储 | SQLite | 本地 |
| 消息队列 | Celery (Redis broker) 或 ARQ | 本地 |
| 联网搜索 | DuckDuckGo / SerpAPI | 云端 |
| 多模态 | Claude Vision / GPT-4V | 云端 API |
| 容器化 | Docker + docker-compose | 本地 |

---

## 13. 未尽事项 & 后续迭代

以下能力不在本期范围，但架构已预留扩展点：

- **GraphRAG**：知识图谱构建 + 社区摘要 → 检索层新增图检索通路即可
- **Agentic RAG**：ReAct 编排 → 在 core/pipeline.py 中增加 Agent 编排模式
- **权限控制**：当前无区分 → 后续在检索层加权限过滤，在 Session 中增加用户身份
- **多语言支持**：当前只做中文 → 增加多语言 embedding + 语言分库路由
- **细粒度权限（文档/段落级）** → 在 Chunk/Session 中增加权限字段
- **用户画像/个性化** → 查询理解层增加用户画像输入
- **监测告警**：检索命中率/响应延迟/幻觉率仪表盘 + 漂移检测告警
