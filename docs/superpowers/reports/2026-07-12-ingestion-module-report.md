# Ingestion 模块 — 开发完成报告

> **日期**: 2026-07-12
> **状态**: 全部实现完成，37 tests pass
> **设计文档**: `docs/superpowers/specs/2026-07-12-ingestion-module-design.md`
> **实施计划**: `docs/superpowers/plans/2026-07-12-ingestion-module.md`

---

## 1. 模块定位

离线文档处理 Pipeline，负责将用户上传的文档转换为可检索的 FAISS 索引。与在线问答 Pipeline 完全分离，互不阻塞。

```
文档文件 → 解析 → 语义分块 → Embedding → FAISS 索引写入
```

---

## 2. 架构

```
文档文件
   │
   ▼
┌─────────────────────────────────────────────┐
│              IngestionPipeline                │
│                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Parser   │──▶│ Chunker  │──▶│Embedder  │ │
│  │  Stage    │   │  Stage   │   │  Stage   │ │
│  └──────────┘   └──────────┘   └──────────┘ │
│       │              │              │        │
│       ▼              ▼              ▼        │
│  ┌──────────────────────────────────────┐   │
│  │         FAISS IndexWriter            │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  [预留] MultiModalStage                     │
│  [预留] ContextualRetrievalStage            │
└─────────────────────────────────────────────┘
```

### 核心抽象

| 组件 | 职责 |
|------|------|
| `Stage` | Protocol — `name: str`, `fatal: bool`, `async run(ctx) -> PipelineContext` |
| `IngestionPipeline` | 编排器，持有 Stage 列表，依次调用，记录耗时和状态 |
| `PipelineContext` | 贯穿全链路的数据容器，携带 document、chunks、errors、metadata |

### 设计原则

- Pipeline 不关心 Stage 内部实现，只调用 `stage.run(ctx)`
- 每个 Stage 从 ctx 读取输入、写回输出
- `fatal=True` 的 Stage 出错时中断 pipeline，`fatal=False` 记录后继续

---

## 3. 目录结构

```
src/ingestion/
├── __init__.py          # 导出 + create_default_pipeline() 工厂函数
├── pipeline.py          # IngestionPipeline 编排器
├── stage.py             # Stage 协议
├── context.py           # PipelineContext + 内置 Document/Chunk 模型
├── parser.py            # ParserStage — PDF/Word/Markdown → Markdown（基于 docling）
├── chunker.py           # ChunkerStage — 语义/固定/层级 三种分块策略
├── embedder.py          # EmbedderStage — 批量 embedding（幂等）
├── indexer.py           # FAISSIndexWriter — FAISS 索引写入
├── [预留] multimodal.py
└── [预留] contextual.py

tests/unit/ingestion/
├── __init__.py
├── test_context.py      # 8 tests — 数据模型
├── test_stage.py        # 3 tests — Stage 协议
├── test_pipeline.py     # 5 tests — 编排器
├── test_parser.py       # 3 tests — 文档解析
├── test_chunker.py      # 9 tests — 分块策略
├── test_embedder.py     # 5 tests — 批量 embedding
└── test_indexer.py      # 4 tests — FAISS 索引
```

---

## 4. 各模块详情

### 4.1 ParserStage

- **解析库**: docling（IBM 开源，MIT 协议）
- **支持格式**: PDF / Word / Markdown（HTML 预留）
- **输出**: 统一 Markdown 格式，表格自动结构化
- **fatal**: `True`（解析失败中断 pipeline）

```python
from ingestion.parser import ParserStage

stage = ParserStage()
ctx = await stage.run(ctx)
# ctx.document.raw_text → Markdown 文本
# ctx.document.metadata → {source_path, file_size, ...}
```

### 4.2 ChunkerStage

三种分块策略，通过 `settings.chunking.strategy` 切换：

| 策略 | 机制 | 适用场景 |
|------|------|----------|
| `semantic` | SemanticChunker — embedding 相似度检测语义边界 | 通用文档（默认） |
| `fixed` | 固定大小 + 滑动窗口重叠 | 无标题的文档 |
| `hierarchical` | 按标题层级 + 父子 chunk | 有清晰 H1-H3 的文档 |

```python
from ingestion.chunker import ChunkerStage

stage = ChunkerStage(embedding_model=model)  # SemanticChunker 需要 embedding
ctx = await stage.run(ctx)
# ctx.chunks → list[Chunk]，带双向链表（prev_chunk_id / next_chunk_id）
```

**SemanticChunker 工作流**:

1. 拆分为句子序列
2. 批量计算句子 embedding（与 EmbedderStage 共享模型实例）
3. 余弦相似度 → percentile 阈值检测语义边界
4. buffer 机制避免过度切分
5. 合并 + 滑动窗口重叠 + 双向链表

### 4.3 EmbedderStage

- **模型**: SentenceTransformer（BGE-large-zh-v1.5）
- **幂等**: 已有 embedding 的 chunk 自动跳过
- **批量**: 按 `settings.embedding.batch_size` 分批
- **fatal**: `False`

```python
from ingestion.embedder import EmbedderStage

stage = EmbedderStage(embedding_model=model)
ctx = await stage.run(ctx)
# ctx.chunks[i].embedding → list[float]（1024 维）
```

### 4.4 FAISSIndexWriter

- **索引类型**: IVF_FLAT（可配置）
- **度量**: COSINE（L2 normalize + Inner Product）
- **存储结构**: `{index_dir}/{collection}/index.faiss` + `docstore.json`
- **增量追加**: collection 存在时追加新 chunk

```
data/faiss_indexes/
├── default/
│   ├── index.faiss
│   └── docstore.json      # chunk_id → {faiss_id, text, doc_id, ...}
├── tech/
│   └── ...
└── policy/
    └── ...
```

### 4.5 IngestionPipeline

```python
from ingestion import create_default_pipeline

pipeline = create_default_pipeline()
ctx = await pipeline.run(Path("document.pdf"), collection="default")

print(ctx.status)        # "done" | "failed"
print(len(ctx.chunks))   # 分块数量
print(ctx.metadata)      # {parser_ms, chunker_ms, embedder_ms, ...}
```

---

## 5. 数据模型

### Document

```python
@dataclass
class Document:
    doc_id: str                         # uuid4
    source_path: Path                   # 文件路径
    file_type: str                      # pdf / docx / md
    title: str = ""                     # 默认取文件名
    raw_text: str = ""                  # Parser 输出
    collection: str = "default"
    metadata: dict = {}
```

### Chunk

```python
@dataclass
class Chunk:
    chunk_id: str                       # uuid4
    doc_id: str                         # 所属文档
    text: str                           # 分块文本
    chunk_index: int                    # 序号（0-based）
    prev_chunk_id: str | None           # 双向链表 ←
    next_chunk_id: str | None           # 双向链表 →
    context_summary: str | None         # Contextual Retrieval 预留
    embedding: list[float] | None       # Embedder 填充
    metadata: dict = {}
```

### PipelineContext

```python
@dataclass
class PipelineContext:
    document: Document
    chunks: list[Chunk] = []
    current_stage: str = ""
    status: str = "pending"             # pending → running → done / failed
    errors: list[StageError] = []
    metadata: dict = {}                 # 各阶段耗时等
```

---

## 6. 配置依赖

```yaml
# 分块
chunking:
  chunk_size: 512
  overlap: 64
  strategy: semantic
  semantic_threshold_percentile: 0.9    # SemanticChunker 切分敏感度
  semantic_buffer_size: 1               # 切分点缓冲区句子数

# Embedding
embedding:
  model: BAAI/bge-large-zh-v1.5
  device: cpu
  batch_size: 32
  dimension: 1024

# FAISS
faiss:
  index_type: IVF_FLAT
  metric_type: COSINE
  nlist: 100
  nprobe: 10
  dimension: 1024
  index_dir: data/faiss_indexes
```

---

## 7. 测试覆盖

| 模块 | 测试数 | 覆盖场景 |
|------|--------|----------|
| context | 8 | 最小/完整构造、链表、默认值、错误标记 |
| stage | 3 | 结构子类型、上下文修改、非 fatal 错误累积 |
| pipeline | 5 | 成功流程、fatal 中断、非 fatal 继续、耗时记录、Document 构造 |
| parser | 3 | Markdown 解析、不存在文件异常、元数据验证 |
| chunker | 9 | Fixed 分块/短文本/链表、Semantic 分块/短文本、Hierarchical 标题、策略选择/空文本/元数据 |
| embedder | 5 | 全部 embedding、幂等跳过、空列表、元数据记录、元数据验证 |
| indexer | 4 | 索引创建、docstore 内容、维度不匹配、增量追加 |
| **总计** | **37** | |

---

## 8. Git 提交记录

```
546bca5 chore: update ingestion deps (docling), add ingestion to known-first-party
8f149a5 feat(ingestion): add create_default_pipeline factory + ChunkingConfig fields
ddc8444 feat(ingestion): add FAISSIndexWriter for index persistence
84be690 feat(ingestion): add EmbedderStage for batch embedding
281030d feat(ingestion): add ChunkerStage with fixed/semantic/hierarchical strategies
d411280 feat(ingestion): add ParserStage with docling support
ba7de3f feat(ingestion): add IngestionPipeline orchestrator
5c546ee feat(ingestion): add Stage protocol
b68ab3a feat(ingestion): add data models — Document, Chunk, PipelineContext, StageError
```

---

## 9. 未尽事项（后续迭代）

以下不在第一期范围，但架构已预留扩展点：

- **MultiModalStage**: 图片 OCR + 多模态描述
- **ContextualRetrievalStage**: chunk embedding 前拼接文档上下文摘要
- **异步队列**: Celery/ARQ 异步处理
- **增量去重**: 同一文档重新上传时替换旧 chunks
- **HTML 解析**: docling 已支持，第一期如需可直接启用
- **更多分块策略**: Late Chunking、Small-to-Big 检索
