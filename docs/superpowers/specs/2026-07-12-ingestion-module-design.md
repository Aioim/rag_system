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
| `Stage` | 协议/基类，定义 `async def run(self, ctx: PipelineContext) -> PipelineContext` |
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
├── parser.py            # ParserStage — PDF/Word/Markdown → 纯文本 + 表格结构化
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
| `application/pdf` | PyMuPDF (fitz) | P0 |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | python-docx | P0 |
| `text/markdown` | stdlib + 正则去标记 | P0 |
| `text/html` | BeautifulSoup → 去标签 + 保留结构 | P1（预留） |

### 输入/输出

```
输入: ctx.document.source_path  (Path)
      ctx.document.file_type     (str)
输出: ctx.document.raw_text      (str)
      ctx.document.pages         (list[Page] | None) — PDF 保留页码信息
      ctx.document.metadata      (dict) — 标题、作者、页数等
```

### 设计决策

- **统一输出为纯文本**，结构信息用标记保留（如 `## 标题`、`| 表格行 |`）
- **表格处理（第一期）**：PDF 表格提取为 Markdown 表格格式；Word 表格同样转换；图片不处理
- 解析失败记录到 `ctx.errors`，不中断 pipeline（标记 `fatal=True`，下游判断 `raw_text` 是否为空）
- 通过文件扩展名 + MIME 双重检测格式，不一致时以 MIME 为准

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

- 三种 splitter 同接口：`split(text: str) -> list[Chunk]`，ChunkerStage 不关心具体策略
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

- 不在此 Stage 加载模型——通过 `models.get_path("embedding")` 获取路径，模型加载/缓存由 model manager 负责
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
    │        - index.id_map.json    → chunk_id → FAISS id 映射
    │        - docstore.json        → chunk_id → 元数据
    └── 5. 写回 ctx.metadata.index_path
```

### 索引存储结构

```
data/faiss_indexes/
├── default/
│   ├── index.faiss
│   ├── index.id_map.json
│   └── docstore.json
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
- docstore.json 存储 chunk 元数据，检索时和向量结果一起返回

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
    file_type: str                      # pdf / docx / md
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
        # 遍历 stages，逐个调用 stage.run(ctx)
        # 记录每阶段耗时到 ctx.metadata
        # fatal 错误时停止，非 fatal 记录后继续
        # 最后调用 index_writer.write(ctx.chunks, collection)
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
    return IngestionPipeline(
        stages=[
            ParserStage(),
            ChunkerStage(),         # 根据 settings.chunking.strategy 选择 splitter
            EmbedderStage(),
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
- **HTML 解析**：BeautifulSoup 解析器（Parser 已预留入口）
- **更多分块策略**：Late Chunking、Small-to-Big 检索（Chunker 的三策略接口已支持扩展）
