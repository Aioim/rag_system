# Ingestion 模块 — 设计文档

> 日期：2026-07-12
> 状态：设计完成，待评审
> 依赖：config / model / logger 模块（已完成）

---

## 1. 模块定位

离线文档处理 Pipeline，负责将用户上传的文档转换为可检索的 FAISS 索引。与在线问答 Pipeline 完全分离，互不阻塞。

```
文档文件 → 解析 → 语义分块 → Embedding → FAISS 索引写入
```

---

## 2. 架构概览

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
| `Stage` | 协议/基类，`name: str`、`fatal: bool`、`async def run(self, ctx: PipelineContext) -> PipelineContext` |
| `IngestionPipeline` | 编排器，持有 Stage 列表，依次调用，记录耗时和状态 |
| `PipelineContext` | 贯穿全链路的数据容器，携带 document、chunks、errors、metadata |

### 设计原则

- Pipeline 不关心 Stage 内部实现，只调用 `stage.run(ctx)`
- 每个 Stage 从 ctx 读取输入、写回输出
- 错误不总是中断链路——fatal 错误停止下游，非 fatal 错误记录到 `ctx.errors` 后继续

---

## 3. 目录结构

```
src/ingestion/
├── __init__.py          # 导出 + create_default_pipeline() 工厂函数
├── pipeline.py          # IngestionPipeline 编排器
├── stage.py             # Stage 基类 / 协议
├── context.py           # PipelineContext + 内置 Document/Chunk 模型
├── parser.py            # ParserStage — PDF/Word/Markdown → Markdown（基于 docling）
├── chunker.py           # ChunkerStage — 语义/固定/层级 三种分块策略
├── embedder.py          # EmbedderStage — 调用 model manager 批量 embedding
├── indexer.py           # FAISSIndexWriter — FAISS 索引写入
├── [预留] multimodal.py
└── [预留] contextual.py
```

Data models（Document、Chunk）先内置在 `context.py`，后续提取到 `src/models/`。

---

## 4. Parser Stage

### 职责

多格式文档 → 统一纯文本 + 基础元数据

### 支持格式

| 格式 | 解析库 | 优先级 |
|------|--------|--------|
| `application/pdf` | docling | P0 |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | docling | P0 |
| `text/markdown` | docling（或 stdlib 直读） | P0 |
| `text/html` | docling | P1（预留） |

### 为什么选 docling

- IBM 开源（MIT 协议），LF AI & Data 基金会托管
- 统一 API 处理 PDF/Word/PPT/HTML/Markdown 等多种格式
- 内置布局分析（DocLayNet）、表格提取（TableFormer）、阅读顺序检测
- 输出统一为 Markdown/JSON，自带表格结构化
- 完全本地运行，无需云服务
- 一期不再需要 PyMuPDF + python-docx 双库，减少依赖

### 输入/输出

```
输入: ctx.document.source_path  (Path)
      ctx.document.file_type     (str)
输出: ctx.document.raw_text      (str) — Markdown 格式
      ctx.document.metadata      (dict) — 标题、作者、页数等
```

### 设计决策

- **统一输出为 Markdown**，docling 原生支持，结构信息自然保留
- **表格处理**：docling 内置 TableFormer 模型，PDF/Word 表格自动转为 Markdown 表格
- **图片处理（第一期不处理）**：docling 支持 VLM 图片描述（GraniteDocling），但不启用，图片仅保留占位标记
- 解析失败记录到 `ctx.errors`，中断 pipeline（标记 `fatal=True`，下游判断 `raw_text` 是否为空）
- 通过 docling 自动检测格式，无需手动 MIME 判断

---

## 5. Chunker Stage

### 职责

纯文本 → 语义分块列表（带双向链表）

### 三种分块策略

| 策略 | 机制 | 适用场景 |
|------|------|----------|
| `semantic` | **SemanticChunker** — embedding 相似度检测语义边界 | 通用文档、长文（默认） |
| `fixed` | 固定大小 + 滑动窗口重叠 | 结构化弱或无标题的文档 |
| `hierarchical` | 按标题层级 + 父子 chunk 结构 | 有清晰 H1-H3 层级的规范文档 |

### SemanticChunker 工作流

```
输入文本
  │
  ├── 1. 按标点/换行拆分为句子序列
  ├── 2. 相邻句子对计算 embedding 余弦相似度（批量调用 embedding 模型）
  ├── 3. 相似度低于阈值的点 → 语义边界，在此切分
  │       - 阈值策略: percentile（取所有相似度的 P 分位）
  │       - 可配置: settings.chunking.semantic_threshold_percentile (默认 0.9)
  │       - buffer: 切分点前后各保留 semantic_buffer_size 个句子，避免在轻微语义转折处过度切分
  ├── 4. 合并：将边界间的句子拼接，控制在 chunk_size 附近
  ├── 5. 滑动窗口重叠 (overlap tokens)
  └── 6. 构建双向链表 (prev_chunk_id / next_chunk_id)
```

### 配置项

```yaml
chunking:
  chunk_size: 512
  overlap: 64
  strategy: semantic                    # semantic | fixed | hierarchical
  semantic_threshold_percentile: 0.9   # SemanticChunker 专用
  semantic_buffer_size: 1              # 切分点缓冲区句子数
```

### 输入/输出

```
输入: ctx.document.raw_text
      settings.chunking.*
输出: ctx.chunks: list[Chunk]
      每个 Chunk 包含:
        - chunk_id       (uuid)
        - doc_id         (来自 ctx.document.doc_id)
        - text           (分块文本)
        - chunk_index    (序号，0-based)
        - prev_chunk_id
        - next_chunk_id
        - context_summary (None — ContextualRetrievalStage 预留)
        - embedding      (None — EmbedderStage 预留)
```

### 设计决策

- 三种 splitter 同接口：`splitter(text: str) -> list[Chunk]`（embedding_model 等外部依赖在构造函数注入，不在方法签名中），ChunkerStage 不关心具体策略
- SemanticChunker 需要 embedding 模型计算句子相似度，模型实例由工厂函数传入，与 EmbedderStage **共享同一实例**（避免 BGE-large ~1.3GB 双倍内存）
- ChunkerStage 根据 `settings.chunking.strategy` 选择 splitter 实例
- Token 计数用中文字数估算（1 字 ≈ 1 token），一期不引入 tokenizer
- 重叠按 token 数计算而非字符数
- 分块为空时记录 error

### 边界处理

| 场景 | 处理方式 |
|------|----------|
| 空文档 | `ctx.errors` 记录，`ctx.chunks = []` |
| 极短文档（< chunk_size） | 生成 1 个 chunk，无前后兄弟 |
| 单段超长（无句子边界） | 退化为固定大小切分 |
| 代码块/表格 | 检测后尽量保持完整，不从中切断 |

---

## 6. Embedder Stage

### 职责

chunk 列表 → 批量 embedding → 写回 chunk

### 核心流程

```
ctx.chunks (无 embedding)
    │
    ├── 1. 过滤：跳过已有 embedding 的 chunk（幂等，支持重跑）
    ├── 2. 批量调用 model manager
    │        models.get_path("embedding")  → 本地模型路径
    │        batch_size = settings.embedding.batch_size (32)
    ├── 3. 结果写回 ctx.chunks[i].embedding
    └── 4. 记录耗时、batch 数量到 ctx.metadata
```

### 输入/输出

```
输入: ctx.chunks: list[Chunk]
      settings.embedding.batch_size      (32)
      settings.embedding.dimension       (1024)

输出: ctx.chunks[i].embedding: list[float]   (1024 维)
      ctx.metadata.embedding_batches: int
      ctx.metadata.embedding_duration_ms: int
```

### 设计决策

- 不在此 Stage 加载模型——通过 `models.get_path("embedding")` 获取路径，模型加载/缓存由 model manager 负责。实际模型实例由工厂函数传入，与 ChunkerStage（SemanticChunker）**共享同一实例**
- **模型调用接口**：`model.encode(texts: list[str]) -> list[ndarray]`（SentenceTransformer 原生接口），SemanticChunker 和 Embedder 都通过此接口调用
- 批量推理：chunks 按 batch_size 分批
- 幂等：已带 embedding 的 chunk 自动跳过
- 空 chunks 时跳过，记录 info 日志
- GPU/CPU 跟随 `settings.embedding.device` 配置

### 依赖

```
EmbedderStage
    ├── model manager (models.get_path / models.load)
    └── settings.embedding (batch_size, dimension, device)
```

---

## 7. FAISS IndexWriter

### 职责

带 embedding 的 chunks → FAISS 索引持久化

### 核心流程

```
ctx.chunks (有 embedding)
    │
    ├── 1. 构建 docstore — chunk_id → chunk 元数据
    ├── 2. 构建 FAISS 索引
    │        - 类型: settings.faiss.index_type (IVF_FLAT)
    │        - 度量: settings.faiss.metric_type (COSINE)
    │        - 维度: settings.faiss.dimension (1024)
    │        - 训练: IVF 训练 nlist (100) 个聚类中心
    ├── 3. 添加向量 + 训练（首次）/ 追加（增量）
    ├── 4. 持久化
    │        - index.faiss         → FAISS 索引二进制
    │        - docstore.json        → chunk_id → {faiss_id, text, doc_id, ...}（与 id 映射合并为一个文件）
    └── 5. 写回 ctx.metadata.index_path
```

### 索引存储结构

```
data/faiss_indexes/
├── default/
│   ├── index.faiss
│   └── docstore.json      # chunk_id → {faiss_id, text, doc_id, chunk_index, ...}
├── tech/
│   └── ...
└── policy/
    └── ...
```

按 collection 隔离，一个 collection = 一个目录 = 一个独立 FAISS 索引。

### 设计决策

- `IndexWriter` 是独立写入器，不是 Pipeline Stage（Stage 语义是"转换数据"，IndexWriter 是"持久化副作用"）
- Pipeline 最后一步调用 `IndexWriter.write(chunks, collection)`，由 `IngestionPipeline` 编排
- **增量追加 vs 全量重建**：collection 不存在→创建；存在→追加。FAISS IVF 不支持真正 delete，一期跳过去重
- 维度校验：写入前校验 embedding 维度与 `settings.faiss.dimension` 一致，不一致直接报错
- `docstore.json` 合并 id 映射和元数据：`{chunk_id: {faiss_id: int, text: str, doc_id: str, ...}}`。FAISS 内部使用自增整数 ID，检索时通过 docstore 将 faiss_id 映射回 chunk 信息

### 依赖

```
FAISSIndexWriter
    ├── faiss-cpu (或 faiss-gpu)
    ├── settings.faiss (index_type, metric_type, nlist, dimension, index_dir)
    └── ctx.chunks (必须有 embedding)
```

---

## 8. Pipeline 编排 & 上下文容器

### PipelineContext

```python
@dataclass
class PipelineContext:
    # 输入
    document: Document

    # 各阶段产出
    chunks: list[Chunk] = field(default_factory=list)

    # 状态追踪
    current_stage: str = ""
    status: str = "pending"             # pending → running → done / failed
    errors: list[StageError] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

### Document & Chunk（内置，后续提取到 src/models/）

```python
@dataclass
class Document:
    doc_id: str
    source_path: Path
    file_type: str                      # 源文件扩展名，第一期启用: pdf / docx / md
    title: str = ""
    raw_text: str = ""
    collection: str = "default"
    metadata: dict = field(default_factory=dict)

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    context_summary: str | None = None  # Contextual Retrieval 预留
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)

@dataclass
class StageError:
    stage: str
    error: str
    fatal: bool = False
```

### IngestionPipeline

```python
class IngestionPipeline:
    def __init__(self, stages: list[Stage], index_writer: IndexWriter):
        ...

    async def run(self, file_path: Path, collection: str = "default") -> PipelineContext:
        # 1. 构造 Document — doc_id 用 uuid4，title 取文件名，file_type 取扩展名
        doc = Document(
            doc_id=str(uuid.uuid4()),
            source_path=file_path,
            file_type=file_path.suffix.lstrip(".").lower(),
            title=file_path.stem,
            collection=collection,
        )
        ctx = PipelineContext(document=doc, status="running")

        # 2. 遍历 stages，逐个调用 stage.run(ctx)
        #    - 记录每阶段耗时到 ctx.metadata
        #    - fatal 错误时停止，非 fatal 记录后继续
        #    - 更新 ctx.current_stage

        # 3. 最后调用 index_writer.write(ctx.chunks, collection)
        # 4. 设置 ctx.status = "done" 并返回
```

### 错误分级

| 级别 | 行为 | 示例 |
|------|------|------|
| `fatal=True` | 停止 pipeline，下游不执行 | Parser 无法打开文件 |
| `fatal=False` | 记录 error，继续执行 | Embedder 部分 batch 失败 |

### 默认管道组装

```python
# ingestion/__init__.py
def create_default_pipeline() -> IngestionPipeline:
    from sentence_transformers import SentenceTransformer

    # 通过 model manager 获取本地路径，再用 SentenceTransformer 加载
    model_path = models.get_path("embedding")
    if model_path is None:
        raise RuntimeError("Embedding 模型未下载，请先运行 models.download('embedding')")
    embedding_model = SentenceTransformer(str(model_path))  # encode(texts) → list[ndarray]

    return IngestionPipeline(
        stages=[
            ParserStage(),
            ChunkerStage(embedding_model=embedding_model),    # 根据 settings.chunking.strategy 选择 splitter
            EmbedderStage(embedding_model=embedding_model),   # 共享同一模型实例
        ],
        index_writer=FAISSIndexWriter(),
    )
```

---

## 9. 配置依赖

本模块读取的配置项（已在 `config/defaults.yaml` 中定义）：

```yaml
# 分块
chunking:
  chunk_size: 512
  overlap: 64
  strategy: semantic
  semantic_threshold_percentile: 0.9   # 新增：SemanticChunker 专用

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

# 模型
model:
  cache_dir: models
  default_models:
    embedding: BAAI/bge-large-zh-v1.5
```

需要新增的配置项：`chunking.semantic_threshold_percentile` 和 `chunking.semantic_buffer_size`。

---

## 10. 未尽事项 & 后续迭代

以下不在第一期范围，但架构已预留扩展点：

- **MultiModalStage**：图片 OCR + 多模态描述（架构已预留 `src/ingestion/multimodal.py`）
- **ContextualRetrievalStage**：chunk embedding 前拼接文档上下文摘要（架构已预留 `contextual.py`，Chunk 已预留 `context_summary` 字段）
- **异步队列**：Celery/ARQ 异步处理（当前同步接口，后续 `IngestionPipeline.run()` 可包装为异步任务）
- **增量去重**：同一文档重新上传时，检测并替换旧 chunks（FAISS IVF 不支持原生 delete，需额外实现）
- **HTML 解析**：docling 已支持 HTML，第一期如需可直接启用
- **更多分块策略**：Late Chunking、Small-to-Big 检索（Chunker 的三策略接口已支持扩展）
