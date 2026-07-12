# Ingestion 模块 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现离线文档处理 Pipeline，将 PDF/Word/Markdown 文档转换为可检索的 FAISS 索引。

**Architecture:** Stage-Pipeline 模式 — ParserStage → ChunkerStage → EmbedderStage → FAISSIndexWriter，通过 PipelineContext 贯穿全链路，工厂函数组装默认管道。

**Tech Stack:** docling（文档解析）、sentence-transformers（Embedding）、faiss-cpu（向量索引）、pytest + pytest-asyncio（测试）

## Global Constraints

- `langchain` >= 1.4.0, `langgraph` >= 1.2.0
- Python >= 3.11
- Ruff line-length 100, double quotes
- 测试目录：`tests/unit/ingestion/`，源文件目录：`src/ingestion/`
- 每个 Stage 的 `fatal` 属性决定错误时是否中断 pipeline
- TDD：先写测试→验证失败→实现→验证通过→提交

## File Map

| 文件 | 职责 |
|------|------|
| `src/ingestion/context.py` | Document, Chunk, PipelineContext, StageError 数据模型 |
| `src/ingestion/stage.py` | Stage 协议（name, fatal, run） |
| `src/ingestion/pipeline.py` | IngestionPipeline 编排器 |
| `src/ingestion/parser.py` | ParserStage — docling 文档解析 |
| `src/ingestion/chunker.py` | ChunkerStage + 三种 splitter（Fixed, Semantic, Hierarchical） |
| `src/ingestion/embedder.py` | EmbedderStage — 批量 embedding |
| `src/ingestion/indexer.py` | FAISSIndexWriter — FAISS 索引持久化 |
| `src/ingestion/__init__.py` | create_default_pipeline() 工厂函数 |
| `pyproject.toml` | 更新 ingestion 依赖（docling 替换 PyMuPDF+python-docx） |

---

### Task 1: Data Models (context.py)

**Files:**
- Create: `tests/unit/ingestion/__init__.py`
- Create: `tests/unit/ingestion/test_context.py`
- Create: `src/ingestion/__init__.py`（空文件，标记为 package）
- Create: `src/ingestion/context.py`

**Interfaces:**
- Produces:
  - `Document(doc_id: str, source_path: Path, file_type: str, title: str = "", raw_text: str = "", collection: str = "default", metadata: dict = {})`
  - `Chunk(chunk_id: str, doc_id: str, text: str, chunk_index: int, prev_chunk_id: str | None = None, next_chunk_id: str | None = None, context_summary: str | None = None, embedding: list[float] | None = None, metadata: dict = {})`
  - `PipelineContext(document: Document, chunks: list[Chunk] = [], current_stage: str = "", status: str = "pending", errors: list[StageError] = [], metadata: dict = {})`
  - `StageError(stage: str, error: str, fatal: bool = False)`

- [ ] **Step 1: 创建目录和空 package 文件**

```bash
mkdir -p tests/unit/ingestion
mkdir -p src/ingestion
```

创建 `src/ingestion/__init__.py`（空文件）
创建 `tests/unit/ingestion/__init__.py`（空文件）

- [ ] **Step 2: 编写失败测试 — `tests/unit/ingestion/test_context.py`**

```python
"""PipelineContext、Document、Chunk、StageError 数据模型测试"""

from pathlib import Path

from ingestion.context import Chunk, Document, PipelineContext, StageError


class TestDocument:
    def test_minimal_construction(self):
        doc = Document(
            doc_id="doc-001",
            source_path=Path("/tmp/test.pdf"),
            file_type="pdf",
        )
        assert doc.doc_id == "doc-001"
        assert doc.file_type == "pdf"
        assert doc.title == ""
        assert doc.raw_text == ""
        assert doc.collection == "default"
        assert doc.metadata == {}

    def test_full_construction(self):
        doc = Document(
            doc_id="doc-002",
            source_path=Path("/tmp/report.docx"),
            file_type="docx",
            title="年度报告",
            raw_text="# 第一章\n内容...",
            collection="tech",
            metadata={"author": "张三", "pages": 10},
        )
        assert doc.title == "年度报告"
        assert doc.collection == "tech"
        assert doc.metadata["author"] == "张三"


class TestChunk:
    def test_minimal_construction(self):
        chunk = Chunk(
            chunk_id="c-001",
            doc_id="doc-001",
            text="这是一段测试文本",
            chunk_index=0,
        )
        assert chunk.chunk_id == "c-001"
        assert chunk.text == "这是一段测试文本"
        assert chunk.prev_chunk_id is None
        assert chunk.next_chunk_id is None
        assert chunk.context_summary is None
        assert chunk.embedding is None
        assert chunk.metadata == {}

    def test_linked_list(self):
        chunk = Chunk(
            chunk_id="c-mid",
            doc_id="d-1",
            text="中间段",
            chunk_index=2,
            prev_chunk_id="c-001",
            next_chunk_id="c-002",
        )
        assert chunk.prev_chunk_id == "c-001"
        assert chunk.next_chunk_id == "c-002"


class TestPipelineContext:
    def test_default_construction(self):
        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        ctx = PipelineContext(document=doc)
        assert ctx.document == doc
        assert ctx.chunks == []
        assert ctx.current_stage == ""
        assert ctx.status == "pending"
        assert ctx.errors == []
        assert ctx.metadata == {}

    def test_with_chunks(self):
        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        chunks = [
            Chunk(chunk_id="c-1", doc_id="d-1", text="A", chunk_index=0),
            Chunk(chunk_id="c-2", doc_id="d-1", text="B", chunk_index=1),
        ]
        ctx = PipelineContext(document=doc, chunks=chunks, status="running")
        assert len(ctx.chunks) == 2
        assert ctx.status == "running"


class TestStageError:
    def test_default_not_fatal(self):
        err = StageError(stage="parser", error="文件无法打开")
        assert err.stage == "parser"
        assert err.error == "文件无法打开"
        assert err.fatal is False

    def test_fatal_error(self):
        err = StageError(stage="parser", error="磁盘满", fatal=True)
        assert err.fatal is True
```

- [ ] **Step 3: 运行测试，验证失败**

```bash
python -m pytest tests/unit/ingestion/test_context.py -v
```
Expected: `ModuleNotFoundError: No module named 'ingestion.context'`

- [ ] **Step 4: 实现 — `src/ingestion/context.py`**

```python
"""Ingestion Pipeline 数据模型 — Document、Chunk、PipelineContext、StageError"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """待处理的文档"""
    doc_id: str
    source_path: Path
    file_type: str                      # 源文件扩展名，第一期启用: pdf / docx / md
    title: str = ""
    raw_text: str = ""
    collection: str = "default"
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """分块后的文本片段"""
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
    """Stage 执行错误"""
    stage: str
    error: str
    fatal: bool = False


@dataclass
class PipelineContext:
    """贯穿全链路的数据容器"""
    document: Document
    chunks: list[Chunk] = field(default_factory=list)
    current_stage: str = ""
    status: str = "pending"             # pending → running → done / failed
    errors: list[StageError] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 5: 运行测试，验证通过**

```bash
python -m pytest tests/unit/ingestion/test_context.py -v
```
Expected: ALL PASS (7 tests)

- [ ] **Step 6: 提交**

```bash
git add tests/unit/ingestion/__init__.py tests/unit/ingestion/test_context.py src/ingestion/__init__.py src/ingestion/context.py
git commit -m "feat(ingestion): add data models — Document, Chunk, PipelineContext, StageError"
```

---

### Task 2: Stage Protocol (stage.py)

**Files:**
- Create: `tests/unit/ingestion/test_stage.py`
- Create: `src/ingestion/stage.py`

**Interfaces:**
- Consumes: `PipelineContext` from `ingestion.context`
- Produces: `Stage` — Protocol with `name: str`, `fatal: bool`, `async def run(self, ctx: PipelineContext) -> PipelineContext`

- [ ] **Step 1: 编写失败测试 — `tests/unit/ingestion/test_stage.py`**

```python
"""Stage 协议测试"""

from pathlib import Path

import pytest

from ingestion.context import Document, PipelineContext
from ingestion.stage import Stage


class FakeParserStage:
    """符合 Stage 协议的示例实现"""
    name = "parser"
    fatal = True

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        ctx.document.raw_text = "parsed content"
        ctx.current_stage = self.name
        return ctx


class FakeChunkerStage:
    """非 fatal 的 Stage"""
    name = "chunker"
    fatal = False

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.document.raw_text:
            from ingestion.context import StageError
            ctx.errors.append(StageError(stage=self.name, error="empty text"))
            return ctx
        ctx.current_stage = self.name
        return ctx


class TestStageProtocol:
    def test_structural_subtyping(self):
        """FakeParserStage 应该通过 Stage 协议的结构类型检查"""
        stage: Stage = FakeParserStage()
        assert stage.name == "parser"
        assert stage.fatal is True

    @pytest.mark.asyncio
    async def test_run_modifies_context(self):
        stage = FakeParserStage()
        doc = Document(doc_id="d-1", source_path=Path("test.pdf"), file_type="pdf")
        ctx = PipelineContext(document=doc)

        result = await stage.run(ctx)
        assert result.document.raw_text == "parsed content"
        assert result.current_stage == "parser"

    @pytest.mark.asyncio
    async def test_non_fatal_error_continues(self):
        stage = FakeChunkerStage()
        doc = Document(doc_id="d-1", source_path=Path("test.pdf"), file_type="pdf")
        ctx = PipelineContext(document=doc)

        result = await stage.run(ctx)
        assert len(result.errors) == 1
        assert result.errors[0].stage == "chunker"
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
python -m pytest tests/unit/ingestion/test_stage.py -v
```
Expected: `ModuleNotFoundError: No module named 'ingestion.stage'`

- [ ] **Step 3: 实现 — `src/ingestion/stage.py`**

```python
"""Stage 协议 — 定义 Pipeline 中每个阶段的接口"""

from typing import Protocol

from ingestion.context import PipelineContext


class Stage(Protocol):
    """Pipeline Stage 协议

    每个 Stage 必须提供 name（标识）、fatal（错误是否中断 pipeline）、
    以及 run(ctx) 方法（接收并返回 PipelineContext）。
    """
    name: str
    fatal: bool

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """执行阶段逻辑，接收 ctx 并返回修改后的 ctx"""
        ...
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
python -m pytest tests/unit/ingestion/test_stage.py -v
```
Expected: ALL PASS (3 tests)

- [ ] **Step 5: 提交**

```bash
git add tests/unit/ingestion/test_stage.py src/ingestion/stage.py
git commit -m "feat(ingestion): add Stage protocol"
```

---

### Task 3: IngestionPipeline (pipeline.py)

**Files:**
- Create: `tests/unit/ingestion/test_pipeline.py`
- Create: `src/ingestion/pipeline.py`

**Interfaces:**
- Consumes: `Stage` from `ingestion.stage`, `PipelineContext`, `Document`, `Chunk`, `StageError` from `ingestion.context`
- Produces: `IngestionPipeline(stages: list[Stage], index_writer)`, `async def run(self, file_path: Path, collection: str = "default") -> PipelineContext`
- Note: `index_writer` 在 Task 7 实现，此处用 mock

- [ ] **Step 1: 编写失败测试 — `tests/unit/ingestion/test_pipeline.py`**

```python
"""IngestionPipeline 编排器测试"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.context import Document, PipelineContext, StageError
from ingestion.pipeline import IngestionPipeline


class FakeFatalStage:
    """模拟 fatal 阶段：首次调用抛出异常"""
    name = "parser"
    fatal = True

    def __init__(self):
        self.call_count = 0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("文件损坏，无法解析")
        ctx.document.raw_text = "parsed"
        return ctx


class FakeNonFatalStage:
    """模拟非 fatal 阶段：出错后继续"""
    name = "embedder"
    fatal = False

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        raise RuntimeError("部分 batch 失败")


class FakeSuccessStage:
    """模拟正常阶段"""
    name = "chunker"
    fatal = False

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        from ingestion.context import Chunk
        ctx.chunks = [
            Chunk(chunk_id="c-1", doc_id=ctx.document.doc_id,
                  text="chunk text", chunk_index=0),
        ]
        return ctx


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        """正常流程：所有 stage 成功 + index_writer 被调用"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeSuccessStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/tmp/test.md"))

        assert ctx.status == "done"
        assert len(ctx.chunks) == 1
        assert ctx.chunks[0].text == "chunk text"
        assert ctx.document.file_type == "md"
        assert ctx.document.title == "test"
        mock_writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_fatal_error_stops_pipeline(self):
        """fatal 错误应立即停止 pipeline，不调用 index_writer"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeFatalStage(), FakeSuccessStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/tmp/bad.pdf"))

        assert ctx.status == "failed"
        assert len(ctx.errors) == 1
        assert ctx.errors[0].stage == "parser"
        assert "文件损坏" in ctx.errors[0].error
        # index_writer 不应被调用
        mock_writer.write.assert_not_called()
        # FakeSuccessStage 不应被执行
        assert ctx.chunks == []

    @pytest.mark.asyncio
    async def test_non_fatal_error_continues(self):
        """非 fatal 错误记录后继续执行"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeSuccessStage(), FakeNonFatalStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/tmp/doc.md"))

        assert ctx.status == "done"
        assert len(ctx.errors) == 1
        assert ctx.errors[0].stage == "embedder"
        assert ctx.errors[0].fatal is False
        mock_writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_records_stage_timing(self):
        """每个 stage 的耗时应记录到 ctx.metadata"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeSuccessStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/tmp/doc.md"))

        assert "chunker_ms" in ctx.metadata
        assert isinstance(ctx.metadata["chunker_ms"], float)

    @pytest.mark.asyncio
    async def test_document_construction(self):
        """验证 Document 由 run() 从 file_path 自动构造"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeSuccessStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/data/年度报告.docx"), collection="tech")

        assert ctx.document.file_type == "docx"
        assert ctx.document.title == "年度报告"
        assert ctx.document.collection == "tech"
        assert ctx.document.source_path == Path("/data/年度报告.docx")
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
python -m pytest tests/unit/ingestion/test_pipeline.py -v
```
Expected: `ModuleNotFoundError: No module named 'ingestion.pipeline'`

- [ ] **Step 3: 实现 — `src/ingestion/pipeline.py`**

```python
"""IngestionPipeline — Stage 编排器"""

import time
import uuid
from pathlib import Path
from typing import Protocol

from ingestion.context import Document, PipelineContext, StageError


class IndexWriter(Protocol):
    """FAISSIndexWriter 协议（避免循环依赖，在 indexer.py 中实现）"""

    def write(self, chunks: list, collection: str) -> None: ...


class IngestionPipeline:
    """离线文档处理 Pipeline 编排器

    依次执行 stages，记录耗时和状态，最后调用 index_writer 持久化。
    """

    def __init__(self, stages: list, index_writer: IndexWriter):
        self.stages = stages
        self.index_writer = index_writer

    async def run(
        self, file_path: Path, collection: str = "default"
    ) -> PipelineContext:
        # 1. 构造 Document
        doc = Document(
            doc_id=str(uuid.uuid4()),
            source_path=file_path,
            file_type=file_path.suffix.lstrip(".").lower(),
            title=file_path.stem,
            collection=collection,
        )
        ctx = PipelineContext(document=doc, status="running")

        # 2. 遍历 stages
        for stage in self.stages:
            ctx.current_stage = stage.name
            t0 = time.perf_counter()
            try:
                ctx = await stage.run(ctx)
            except Exception as e:
                ctx.errors.append(
                    StageError(stage=stage.name, error=str(e), fatal=stage.fatal)
                )
                if stage.fatal:
                    ctx.status = "failed"
                    return ctx
            finally:
                ctx.metadata[f"{stage.name}_ms"] = (
                    time.perf_counter() - t0
                ) * 1000

        # 3. 写入索引
        self.index_writer.write(ctx.chunks, collection)

        # 4. 完成
        ctx.status = "done"
        return ctx
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
python -m pytest tests/unit/ingestion/test_pipeline.py -v
```
Expected: ALL PASS (5 tests)

- [ ] **Step 5: 提交**

```bash
git add tests/unit/ingestion/test_pipeline.py src/ingestion/pipeline.py
git commit -m "feat(ingestion): add IngestionPipeline orchestrator"
```

---

### Task 4: ParserStage (parser.py)

**Files:**
- Create: `tests/unit/ingestion/test_parser.py`
- Create: `src/ingestion/parser.py`

**Interfaces:**
- Consumes: `Stage` from `ingestion.stage`, `PipelineContext` from `ingestion.context`
- Produces: `ParserStage` — `name="parser"`, `fatal=True`, `async run(ctx) -> PipelineContext`（写入 `ctx.document.raw_text`、`ctx.document.metadata`）

- [ ] **Step 1: 编写失败测试 — `tests/unit/ingestion/test_parser.py`**

```python
"""ParserStage 测试"""

import tempfile
from pathlib import Path

import pytest

from ingestion.context import Document, PipelineContext
from ingestion.parser import ParserStage


class TestParserStage:
    @pytest.mark.asyncio
    async def test_parse_markdown_file(self):
        """解析 Markdown 文件"""
        stage = ParserStage()
        doc = Document(
            doc_id="d-md",
            source_path=Path("/dev/null"),  # 会被 ctx 中的覆盖
            file_type="md",
        )
        ctx = PipelineContext(document=doc)

        # 创建临时 markdown 文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# 测试标题\n\n这是一段**测试**内容。\n\n- 列表项 1\n- 列表项 2\n")
            tmp_path = Path(f.name)

        try:
            ctx.document.source_path = tmp_path
            result = await stage.run(ctx)
            assert "# 测试标题" in result.document.raw_text
            assert "测试" in result.document.raw_text
            assert len(result.document.raw_text) > 0
        finally:
            tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self):
        """解析不存在的文件应抛出异常（fatal）"""
        stage = ParserStage()
        doc = Document(
            doc_id="d-bad",
            source_path=Path("/nonexistent/file.pdf"),
            file_type="pdf",
        )
        ctx = PipelineContext(document=doc)

        with pytest.raises(Exception):
            await stage.run(ctx)

    @pytest.mark.asyncio
    async def test_name_and_fatal(self):
        """验证 Stage 元数据"""
        stage = ParserStage()
        assert stage.name == "parser"
        assert stage.fatal is True
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
python -m pytest tests/unit/ingestion/test_parser.py -v
```
Expected: `ModuleNotFoundError: No module named 'ingestion.parser'`

- [ ] **Step 3: 实现 — `src/ingestion/parser.py`**

```python
"""ParserStage — 基于 docling 的多格式文档解析"""

from pathlib import Path

from ingestion.context import PipelineContext


class ParserStage:
    """使用 docling 解析 PDF/Word/Markdown → Markdown 文本"""

    name = "parser"
    fatal = True

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        source_path = ctx.document.source_path

        if not source_path.exists():
            raise FileNotFoundError(f"文件不存在: {source_path}")

        if ctx.document.file_type in ("md", "markdown"):
            # Markdown 文件直接读取，不走 docling
            ctx.document.raw_text = source_path.read_text(encoding="utf-8")
        else:
            # PDF/Word 通过 docling 解析
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(source_path))
            ctx.document.raw_text = result.document.export_to_markdown()

        ctx.document.metadata.setdefault("source_path", str(source_path))
        ctx.document.metadata.setdefault("file_size", source_path.stat().st_size)

        return ctx
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
python -m pytest tests/unit/ingestion/test_parser.py -v
```
Expected: ALL PASS (3 tests)

- [ ] **Step 5: 提交**

```bash
git add tests/unit/ingestion/test_parser.py src/ingestion/parser.py
git commit -m "feat(ingestion): add ParserStage with docling support"
```

---

### Task 5: ChunkerStage + Splitters (chunker.py)

**Files:**
- Create: `tests/unit/ingestion/test_chunker.py`
- Create: `src/ingestion/chunker.py`

**Interfaces:**
- Consumes: `PipelineContext` from `ingestion.context`, `settings` from `config`
- Produces:
  - `FixedChunker(chunk_size: int, overlap: int)` — `splitter(text: str) -> list[Chunk]`
  - `SemanticChunker(embedding_model, chunk_size: int, overlap: int, threshold_percentile: float, buffer_size: int)` — `splitter(text: str) -> list[Chunk]`
  - `HierarchicalChunker(chunk_size: int, overlap: int)` — `splitter(text: str) -> list[Chunk]`
  - `ChunkerStage(embedding_model)` — `name="chunker"`, `fatal=False`, `async run(ctx) -> PipelineContext`

- [ ] **Step 1: 编写失败测试 — `tests/unit/ingestion/test_chunker.py`**

```python
"""ChunkerStage + 三种 splitter 测试"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ingestion.context import Chunk, Document, PipelineContext, StageError
from ingestion.chunker import (
    ChunkerStage,
    FixedChunker,
    HierarchicalChunker,
    SemanticChunker,
)


# ---- 测试用 embedding model mock ----

def _make_mock_embedding_model():
    """创建一个假的 embedding model，返回随机向量"""
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts: [np.random.rand(1024).astype(np.float32)
                                    for _ in texts]
    )
    return model


# ---- FixedChunker ----

class TestFixedChunker:
    def test_basic_split(self):
        splitter = FixedChunker(chunk_size=100, overlap=20)
        text = "这是测试。" * 20  # ~120 字
        chunks = splitter.splitter(text)

        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id
            assert c.doc_id == ""  # doc_id 由 ChunkerStage 填入
            assert len(c.text) > 0

    def test_short_text_single_chunk(self):
        splitter = FixedChunker(chunk_size=500, overlap=50)
        chunks = splitter.splitter("短文本")
        assert len(chunks) == 1

    def test_linked_list_links(self):
        splitter = FixedChunker(chunk_size=80, overlap=20)
        text = "测试内容。" * 50
        chunks = splitter.splitter(text)

        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            if i > 0:
                assert c.prev_chunk_id == chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                assert c.next_chunk_id == chunks[i + 1].chunk_id


# ---- SemanticChunker ----

class TestSemanticChunker:
    def test_basic_split(self):
        model = _make_mock_embedding_model()
        splitter = SemanticChunker(
            embedding_model=model,
            chunk_size=200,
            overlap=30,
            threshold_percentile=0.9,
            buffer_size=1,
        )
        # 构造有明显语义段落变化的文本
        text = (
            "Python 是一种解释型语言。" * 10
            + "\n\n"
            + "机器学习是人工智能的一个分支。" * 10
        )
        chunks = splitter.splitter(text)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_short_text(self):
        model = _make_mock_embedding_model()
        splitter = SemanticChunker(
            embedding_model=model,
            chunk_size=500,
            overlap=30,
        )
        chunks = splitter.splitter("很短的文本")
        assert len(chunks) == 1


# ---- HierarchicalChunker ----

class TestHierarchicalChunker:
    def test_split_by_headings(self):
        splitter = HierarchicalChunker(chunk_size=200, overlap=30)
        text = (
            "# 第一章\n\n这是第一章的内容。" * 5
            + "\n\n## 1.1 小节\n\n这是小节内容。" * 5
            + "\n\n# 第二章\n\n这是第二章的内容。" * 5
        )
        chunks = splitter.splitter(text)

        assert len(chunks) >= 2
        # 标题信息应保留在 metadata 中
        headings_found = any(
            "第一章" in c.metadata.get("heading_path", "") for c in chunks
        )
        assert headings_found


# ---- ChunkerStage ----

class TestChunkerStage:
    @pytest.mark.asyncio
    async def test_selects_fixed_strategy(self):
        """ChunkerStage 根据 settings 选择 splitter"""
        import config.settings as _s

        # 临时切换为 fixed 策略
        original = _s.settings.chunking.strategy
        _s.settings.chunking.strategy = "fixed"

        try:
            stage = ChunkerStage()
            doc = Document(
                doc_id="d-1",
                source_path=Path("test.md"),
                file_type="md",
                raw_text="测试内容。" * 30,
            )
            ctx = PipelineContext(document=doc)
            result = await stage.run(ctx)

            assert len(result.chunks) > 0
            assert all(c.doc_id == "d-1" for c in result.chunks)
        finally:
            _s.settings.chunking.strategy = original

    @pytest.mark.asyncio
    async def test_empty_text_records_error(self):
        stage = ChunkerStage()
        doc = Document(
            doc_id="d-1",
            source_path=Path("test.md"),
            file_type="md",
            raw_text="",
        )
        ctx = PipelineContext(document=doc)
        result = await stage.run(ctx)

        assert result.chunks == []
        assert len(result.errors) >= 1
        assert any("empty" in e.error.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_name_and_fatal(self):
        stage = ChunkerStage()
        assert stage.name == "chunker"
        assert stage.fatal is False
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
python -m pytest tests/unit/ingestion/test_chunker.py -v
```
Expected: `ModuleNotFoundError: No module named 'ingestion.chunker'`

- [ ] **Step 3: 实现 FixedChunker + HierarchicalChunker — `src/ingestion/chunker.py`（第一部分）**

```python
"""ChunkerStage — 语义/固定/层级 三种分块策略"""

import re
import uuid

from ingestion.context import Chunk, PipelineContext, StageError


# ============================================================================
# Splitter 基类
# ============================================================================

class _BaseSplitter:
    """Splitter 基类：提供双向链表构建、token 估算等共用逻辑"""

    def _estimate_tokens(self, text: str) -> int:
        """中文字数估算 token 数（1 字 ≈ 1 token）"""
        return len(text)

    def _build_chunks(self, texts: list[str], doc_id: str = "") -> list[Chunk]:
        """为文本列表生成 Chunk，自动建立双向链表"""
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=text,
                chunk_index=i,
            )
            chunks.append(chunk)

        for i, c in enumerate(chunks):
            if i > 0:
                c.prev_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                c.next_chunk_id = chunks[i + 1].chunk_id

        return chunks


# ============================================================================
# FixedChunker — 固定大小 + 滑动窗口重叠
# ============================================================================

class FixedChunker(_BaseSplitter):
    """固定大小分块，相邻 chunk 有 overlap 重叠"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def splitter(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        step = max(self.chunk_size - self.overlap, 1)
        text_segments = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            text_segments.append(text[start:end])
            if end == len(text):
                break
            start += step

        return self._build_chunks(text_segments)


# ============================================================================
# HierarchicalChunker — 按标题层级分块
# ============================================================================

class HierarchicalChunker(_BaseSplitter):
    """按 Markdown 标题层级分块，保留 heading_path 到 metadata"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def splitter(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        # 按标题分割
        sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)

        heading_stack = []
        text_segments = []
        heading_paths = []

        for section in sections:
            if not section.strip():
                continue

            # 提取标题
            heading_match = re.match(r"^(#{1,3})\s+(.+)", section)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                # 更新标题栈
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading_text)

            heading_path = " > ".join(heading_stack) if heading_stack else ""
            heading_paths.append(heading_path)

            # 按 chunk_size 进一步切分长 section
            content = section
            if len(content) > self.chunk_size:
                for i in range(0, len(content), self.chunk_size - self.overlap):
                    seg = content[i: i + self.chunk_size]
                    if seg.strip():
                        text_segments.append(seg)
                        heading_paths.append(heading_path)
            else:
                text_segments.append(content)

        chunks = self._build_chunks(text_segments)
        for c, hp in zip(chunks, heading_paths):
            c.metadata["heading_path"] = hp

        return chunks
```

- [ ] **Step 4: 实现 SemanticChunker — 接续 `src/ingestion/chunker.py`**

在文件末尾追加：

```python
# ============================================================================
# SemanticChunker — embedding 相似度检测语义边界
# ============================================================================

class SemanticChunker(_BaseSplitter):
    """通过相邻句子 embedding 余弦相似度检测语义边界"""

    def __init__(
        self,
        embedding_model,
        chunk_size: int = 512,
        overlap: int = 64,
        threshold_percentile: float = 0.9,
        buffer_size: int = 1,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.threshold_percentile = threshold_percentile
        self.buffer_size = buffer_size

    def splitter(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        # 1. 拆分为句子
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._build_chunks([text])

        # 2. 批量计算句子 embedding
        embeddings = self.embedding_model.encode(sentences)

        # 3. 计算相邻句子余弦相似度
        import numpy as np

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(float(sim))

        # 4. 取 percentile 作为阈值，低于阈值的点即为语义边界
        threshold = np.percentile(similarities, self.threshold_percentile * 100)

        # 5. 标记切分点（应用 buffer）
        cut_points = set()
        for i, sim in enumerate(similarities):
            if sim < threshold:
                # buffer: 切分点前后各保留 buffer_size 个句子不切
                for offset in range(-self.buffer_size, self.buffer_size + 1):
                    idx = i + offset
                    if 0 <= idx < len(similarities):
                        cut_points.add(idx + 1)

        # 6. 按切分点合并句子
        text_segments = []
        start = 0
        for cut in sorted(cut_points):
            if cut > start:
                seg = "".join(sentences[start:cut])
                if seg.strip():
                    text_segments.append(seg)
                start = cut
        if start < len(sentences):
            seg = "".join(sentences[start:])
            if seg.strip():
                text_segments.append(seg)

        # 7. 合并过短的 segment，确保靠近 chunk_size
        text_segments = self._merge_short_segments(text_segments)

        # 8. 应用滑动窗口重叠
        chunks = self._build_chunks(text_segments)
        if self.overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_end = chunks[i - 1].text[-self.overlap:]
                if prev_end:
                    chunks[i].text = prev_end + chunks[i].text

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """按标点拆分句子"""
        # 中文句末标点：。！？；\n
        # 英文句末标点：. ! ? ;
        raw = re.split(r"(?<=[。！？；\n\.\!\?\;])", text)
        return [s for s in raw if s.strip()]

    def _merge_short_segments(self, segments: list[str]) -> list[str]:
        """合并过短的 segment，控制每个 segment 接近 chunk_size"""
        merged = []
        buffer = ""
        for seg in segments:
            if self._estimate_tokens(buffer + seg) <= self.chunk_size:
                buffer += seg
            else:
                if buffer.strip():
                    merged.append(buffer)
                buffer = seg
        if buffer.strip():
            merged.append(buffer)
        return merged
```

- [ ] **Step 5: 实现 ChunkerStage — 接续 `src/ingestion/chunker.py`**

在文件末尾追加：

```python
# ============================================================================
# ChunkerStage — Pipeline Stage
# ============================================================================

class ChunkerStage:
    """根据 settings.chunking.strategy 选择 splitter 并执行分块"""

    name = "chunker"
    fatal = False

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        from config import settings

        raw_text = ctx.document.raw_text
        if not raw_text or not raw_text.strip():
            ctx.errors.append(
                StageError(stage=self.name, error="empty document text, no chunks")
            )
            return ctx

        cfg = settings.chunking
        strategy = cfg.strategy

        # 选择 splitter
        if strategy == "semantic":
            if self.embedding_model is None:
                raise ValueError(
                    "SemanticChunker 需要 embedding_model，请通过 ChunkerStage(embedding_model=...) 传入"
                )
            splitter = SemanticChunker(
                embedding_model=self.embedding_model,
                chunk_size=cfg.chunk_size,
                overlap=cfg.overlap,
                threshold_percentile=getattr(cfg, "semantic_threshold_percentile", 0.9),
                buffer_size=getattr(cfg, "semantic_buffer_size", 1),
            )
        elif strategy == "fixed":
            splitter = FixedChunker(
                chunk_size=cfg.chunk_size,
                overlap=cfg.overlap,
            )
        elif strategy == "hierarchical":
            splitter = HierarchicalChunker(
                chunk_size=cfg.chunk_size,
                overlap=cfg.overlap,
            )
        else:
            ctx.errors.append(
                StageError(
                    stage=self.name,
                    error=f"未知分块策略: {strategy}，可选: semantic | fixed | hierarchical",
                )
            )
            return ctx

        # 执行分块
        chunks = splitter.splitter(raw_text)

        # 回填 doc_id
        for c in chunks:
            c.doc_id = ctx.document.doc_id

        ctx.chunks = chunks

        if not chunks:
            ctx.errors.append(
                StageError(stage=self.name, error="chunking produced zero chunks")
            )

        return ctx
```

- [ ] **Step 6: 运行测试，验证通过**

```bash
python -m pytest tests/unit/ingestion/test_chunker.py -v
```
Expected: ALL PASS (10 tests)

- [ ] **Step 7: 提交**

```bash
git add tests/unit/ingestion/test_chunker.py src/ingestion/chunker.py
git commit -m "feat(ingestion): add ChunkerStage with fixed/semantic/hierarchical strategies"
```

---

### Task 6: EmbedderStage (embedder.py)

**Files:**
- Create: `tests/unit/ingestion/test_embedder.py`
- Create: `src/ingestion/embedder.py`

**Interfaces:**
- Consumes: `PipelineContext`, `Chunk` from `ingestion.context`
- Produces: `EmbedderStage(embedding_model)` — `name="embedder"`, `fatal=False`, `async run(ctx) -> PipelineContext`

- [ ] **Step 1: 编写失败测试 — `tests/unit/ingestion/test_embedder.py`**

```python
"""EmbedderStage 测试"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ingestion.context import Chunk, Document, PipelineContext
from ingestion.embedder import EmbedderStage


def _make_mock_embedding_model():
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts: [np.random.rand(1024).astype(np.float32)
                                    for _ in texts]
    )
    return model


class TestEmbedderStage:
    @pytest.mark.asyncio
    async def test_embeds_all_chunks(self):
        model = _make_mock_embedding_model()
        stage = EmbedderStage(embedding_model=model)

        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        chunks = [
            Chunk(chunk_id="c-1", doc_id="d-1", text="文本 A", chunk_index=0),
            Chunk(chunk_id="c-2", doc_id="d-1", text="文本 B", chunk_index=1),
            Chunk(chunk_id="c-3", doc_id="d-1", text="文本 C", chunk_index=2),
        ]
        ctx = PipelineContext(document=doc, chunks=chunks)

        result = await stage.run(ctx)

        for c in result.chunks:
            assert c.embedding is not None
            assert len(c.embedding) == 1024

    @pytest.mark.asyncio
    async def test_skips_already_embedded_chunks(self):
        """已有 embedding 的 chunk 应跳过（幂等）"""
        model = _make_mock_embedding_model()
        stage = EmbedderStage(embedding_model=model)

        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        pre_embedded = Chunk(
            chunk_id="c-done", doc_id="d-1", text="已处理",
            chunk_index=0, embedding=[0.5] * 1024,
        )
        new_chunk = Chunk(
            chunk_id="c-new", doc_id="d-1", text="新数据",
            chunk_index=1,
        )
        ctx = PipelineContext(document=doc, chunks=[pre_embedded, new_chunk])

        result = await stage.run(ctx)

        # pre_embedded 的 embedding 不应被覆盖
        assert result.chunks[0].embedding == [0.5] * 1024
        # new_chunk 应有新的 embedding
        assert result.chunks[1].embedding is not None

        # model.encode 应该只被调用一次（仅 new_chunk）
        assert model.encode.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_chunks_skips(self):
        """chunks 为空时跳过，不报错"""
        model = _make_mock_embedding_model()
        stage = EmbedderStage(embedding_model=model)

        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        ctx = PipelineContext(document=doc, chunks=[])

        result = await stage.run(ctx)
        assert result.chunks == []
        model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_metadata(self):
        model = _make_mock_embedding_model()
        stage = EmbedderStage(embedding_model=model)

        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        chunks = [
            Chunk(chunk_id="c-1", doc_id="d-1", text="A", chunk_index=0),
        ]
        ctx = PipelineContext(document=doc, chunks=chunks)

        result = await stage.run(ctx)
        assert "embedding_batches" in result.metadata
        assert result.metadata["embedding_batches"] >= 1

    @pytest.mark.asyncio
    async def test_name_and_fatal(self):
        stage = EmbedderStage(embedding_model=_make_mock_embedding_model())
        assert stage.name == "embedder"
        assert stage.fatal is False
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
python -m pytest tests/unit/ingestion/test_embedder.py -v
```
Expected: `ModuleNotFoundError: No module named 'ingestion.embedder'`

- [ ] **Step 3: 实现 — `src/ingestion/embedder.py`**

```python
"""EmbedderStage — 批量 embedding，将向量写回 chunk"""

import time

from ingestion.context import PipelineContext


class EmbedderStage:
    """使用 SentenceTransformer 对 chunk 文本批量编码"""

    name = "embedder"
    fatal = False

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        from config import settings

        chunks = ctx.chunks
        if not chunks:
            return ctx

        # 过滤已 embedding 的 chunk（幂等）
        pending = [c for c in chunks if c.embedding is None]
        if not pending:
            return ctx

        batch_size = settings.embedding.batch_size
        total_batches = 0
        t0 = time.perf_counter()

        for i in range(0, len(pending), batch_size):
            batch = pending[i: i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedding_model.encode(texts)
            for c, emb in zip(batch, embeddings):
                c.embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            total_batches += 1

        ctx.metadata["embedding_batches"] = total_batches
        ctx.metadata["embedding_duration_ms"] = (
            time.perf_counter() - t0
        ) * 1000

        return ctx
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
python -m pytest tests/unit/ingestion/test_embedder.py -v
```
Expected: ALL PASS (5 tests)

- [ ] **Step 5: 提交**

```bash
git add tests/unit/ingestion/test_embedder.py src/ingestion/embedder.py
git commit -m "feat(ingestion): add EmbedderStage for batch embedding"
```

---

### Task 7: FAISSIndexWriter (indexer.py)

**Files:**
- Create: `tests/unit/ingestion/test_indexer.py`
- Create: `src/ingestion/indexer.py`

**Interfaces:**
- Consumes: `Chunk` from `ingestion.context`, `settings` from `config`
- Produces: `FAISSIndexWriter` — `write(self, chunks: list[Chunk], collection: str) -> None`

- [ ] **Step 1: 编写失败测试 — `tests/unit/ingestion/test_indexer.py`**

```python
"""FAISSIndexWriter 测试"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ingestion.context import Chunk
from ingestion.indexer import FAISSIndexWriter


def _make_chunks(n: int = 5, dim: int = 128) -> list[Chunk]:
    """创建带随机 embedding 的测试 chunk"""
    chunks = []
    for i in range(n):
        emb = np.random.rand(dim).astype(np.float32).tolist()
        chunks.append(
            Chunk(
                chunk_id=f"c-{i:03d}",
                doc_id="doc-test",
                text=f"chunk text {i}",
                chunk_index=i,
                embedding=emb,
                metadata={"source": "test"},
            )
        )
    return chunks


class TestFAISSIndexWriter:
    def test_write_creates_index_and_docstore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 临时覆盖 faiss 配置
            import config.settings as _s

            original_dir = _s.settings.faiss.index_dir
            original_dim = _s.settings.faiss.dimension
            _s.settings.faiss.index_dir = tmpdir
            _s.settings.faiss.dimension = 128

            try:
                writer = FAISSIndexWriter()
                chunks = _make_chunks(5)
                writer.write(chunks, "test_collection")

                # 验证文件存在
                idx_dir = Path(tmpdir) / "test_collection"
                assert idx_dir.exists()
                assert (idx_dir / "index.faiss").exists()
                assert (idx_dir / "docstore.json").exists()
            finally:
                _s.settings.faiss.index_dir = original_dir
                _s.settings.faiss.dimension = original_dim

    def test_docstore_contains_chunk_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import config.settings as _s

            original_dir = _s.settings.faiss.index_dir
            original_dim = _s.settings.faiss.dimension
            _s.settings.faiss.index_dir = tmpdir
            _s.settings.faiss.dimension = 128

            try:
                writer = FAISSIndexWriter()
                chunks = _make_chunks(3)
                writer.write(chunks, "test_collection")

                # 读取 docstore，验证内容
                docstore_path = Path(tmpdir) / "test_collection" / "docstore.json"
                with open(docstore_path, encoding="utf-8") as f:
                    docstore = json.load(f)

                assert "c-000" in docstore
                assert docstore["c-000"]["text"] == "chunk text 0"
                assert docstore["c-000"]["doc_id"] == "doc-test"
                assert "faiss_id" in docstore["c-000"]
            finally:
                _s.settings.faiss.index_dir = original_dir
                _s.settings.faiss.dimension = original_dim

    def test_dimension_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import config.settings as _s

            original_dir = _s.settings.faiss.index_dir
            original_dim = _s.settings.faiss.dimension
            _s.settings.faiss.index_dir = tmpdir
            _s.settings.faiss.dimension = 256  # 期望 256 维

            try:
                writer = FAISSIndexWriter()
                chunks = _make_chunks(5, dim=128)  # 实际 128 维
                with pytest.raises(ValueError, match="维度"):
                    writer.write(chunks, "test_collection")
            finally:
                _s.settings.faiss.index_dir = original_dir
                _s.settings.faiss.dimension = original_dim

    def test_append_to_existing_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import config.settings as _s

            original_dir = _s.settings.faiss.index_dir
            original_dim = _s.settings.faiss.dimension
            _s.settings.faiss.index_dir = tmpdir
            _s.settings.faiss.dimension = 128

            try:
                writer = FAISSIndexWriter()

                # 第一批
                batch1 = _make_chunks(3)
                writer.write(batch1, "test_collection")

                # 第二批（追加）
                batch2 = _make_chunks(2)
                writer.write(batch2, "test_collection")

                # 验证 docstore 中有 5 条记录
                docstore_path = Path(tmpdir) / "test_collection" / "docstore.json"
                with open(docstore_path, encoding="utf-8") as f:
                    docstore = json.load(f)
                assert len(docstore) == 5
            finally:
                _s.settings.faiss.index_dir = original_dir
                _s.settings.faiss.dimension = original_dim
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
python -m pytest tests/unit/ingestion/test_indexer.py -v
```
Expected: `ModuleNotFoundError: No module named 'ingestion.indexer'`

- [ ] **Step 3: 实现 — `src/ingestion/indexer.py`**

```python
"""FAISSIndexWriter — FAISS 索引持久化"""

import json
from pathlib import Path

import faiss
import numpy as np


class FAISSIndexWriter:
    """将带 embedding 的 chunks 写入 FAISS 索引 + docstore"""

    def write(self, chunks: list, collection: str) -> None:
        if not chunks:
            return

        from config import settings

        cfg = settings.faiss
        expected_dim = cfg.dimension

        # 维度校验
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.chunk_id} 无 embedding，无法写入索引")
            if len(c.embedding) != expected_dim:
                raise ValueError(
                    f"Chunk {c.chunk_id} embedding 维度 {len(c.embedding)} "
                    f"与配置 {expected_dim} 不一致"
                )

        # 索引目录
        index_dir = Path(cfg.index_dir) / collection
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir / "index.faiss"
        docstore_path = index_dir / "docstore.json"

        # 加载已有 docstore（如果存在）
        existing_docstore = {}
        if docstore_path.exists():
            with open(docstore_path, encoding="utf-8") as f:
                existing_docstore = json.load(f)

        # 构建向量矩阵
        vectors = np.array([c.embedding for c in chunks], dtype=np.float32)

        # 加载或创建 FAISS 索引
        if index_path.exists():
            index = faiss.read_index(str(index_path))
        else:
            dim = expected_dim
            if cfg.index_type == "IVF_FLAT":
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, cfg.nlist)

                if cfg.metric_type == "COSINE":
                    # COSINE 相似度通过 L2 normalize + IP 实现
                    faiss.normalize_L2(vectors)
            else:
                # 默认 FlatIP
                index = faiss.IndexFlatIP(dim)

        # 训练 IVF（需要足够多的向量）
        if isinstance(index, faiss.IndexIVFFlat) and not index.is_trained:
            if len(vectors) >= cfg.nlist:
                index.train(vectors)
            else:
                # 向量不足时降级为 Flat
                index = faiss.IndexFlatIP(expected_dim)

        # COSINE: normalize
        if cfg.metric_type == "COSINE":
            faiss.normalize_L2(vectors)

        # 添加向量
        start_id = index.ntotal
        index.add(vectors)

        # 持久化索引
        faiss.write_index(index, str(index_path))

        # 持久化 docstore（合并已有 + 新 chunk）
        new_entries = {}
        for i, c in enumerate(chunks):
            entry = {
                "faiss_id": start_id + i,
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
            }
            if c.prev_chunk_id:
                entry["prev_chunk_id"] = c.prev_chunk_id
            if c.next_chunk_id:
                entry["next_chunk_id"] = c.next_chunk_id
            if c.metadata:
                entry["metadata"] = c.metadata
            new_entries[c.chunk_id] = entry

        existing_docstore.update(new_entries)
        with open(docstore_path, "w", encoding="utf-8") as f:
            json.dump(existing_docstore, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
python -m pytest tests/unit/ingestion/test_indexer.py -v
```
Expected: ALL PASS (4 tests)

- [ ] **Step 5: 提交**

```bash
git add tests/unit/ingestion/test_indexer.py src/ingestion/indexer.py
git commit -m "feat(ingestion): add FAISSIndexWriter for index persistence"
```

---

### Task 8: Factory Function (__init__.py)

**Files:**
- Modify: `src/ingestion/__init__.py`
- Create: `tests/unit/ingestion/test_init.py`

**Interfaces:**
- Consumes: All modules above
- Produces: `create_default_pipeline() -> IngestionPipeline`

- [ ] **Step 1: 编写失败测试 — `tests/unit/ingestion/test_init.py`**

```python
"""工厂函数 & 集成测试"""

from unittest.mock import MagicMock, patch

import pytest


class TestCreateDefaultPipeline:
    def test_returns_pipeline_with_all_stages(self):
        """工厂函数应返回包含三个 Stage 的 IngestionPipeline"""
        mock_model = MagicMock()

        with patch("ingestion.SentenceTransformer", return_value=mock_model):
            from ingestion import create_default_pipeline

            pipeline = create_default_pipeline()

        assert len(pipeline.stages) == 3
        assert pipeline.stages[0].name == "parser"
        assert pipeline.stages[1].name == "chunker"
        assert pipeline.stages[2].name == "embedder"

    def test_raises_when_model_not_downloaded(self):
        """embedding 模型未下载时应抛出 RuntimeError"""
        with patch("ingestion.models") as mock_models:
            mock_models.get_path.return_value = None
            from ingestion import create_default_pipeline

            with pytest.raises(RuntimeError, match="未下载"):
                create_default_pipeline()
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
python -m pytest tests/unit/ingestion/test_init.py -v
```
Expected: `ImportError` — `create_default_pipeline` 不存在

- [ ] **Step 3: 实现 — 覆盖 `src/ingestion/__init__.py`**

```python
"""Ingestion 模块 — 离线文档处理 Pipeline

使用示例:
    from ingestion import create_default_pipeline

    pipeline = create_default_pipeline()
    ctx = await pipeline.run(Path("document.pdf"), collection="default")
    print(ctx.status, len(ctx.chunks))
"""

from pathlib import Path

from model import models

from .pipeline import IngestionPipeline
from .parser import ParserStage
from .chunker import ChunkerStage
from .embedder import EmbedderStage
from .indexer import FAISSIndexWriter


def create_default_pipeline() -> IngestionPipeline:
    """组装默认的 ingestion pipeline

    加载 embedding 模型一次，Chunker（SemanticChunker）和 Embedder 共享同一实例。
    """
    from sentence_transformers import SentenceTransformer

    model_path = models.get_path("embedding")
    if model_path is None:
        raise RuntimeError(
            "Embedding 模型未下载，请先运行 models.download('embedding')"
        )

    embedding_model = SentenceTransformer(str(model_path))

    return IngestionPipeline(
        stages=[
            ParserStage(),
            ChunkerStage(embedding_model=embedding_model),
            EmbedderStage(embedding_model=embedding_model),
        ],
        index_writer=FAISSIndexWriter(),
    )
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
python -m pytest tests/unit/ingestion/test_init.py -v
```
Expected: ALL PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add tests/unit/ingestion/test_init.py src/ingestion/__init__.py
git commit -m "feat(ingestion): add create_default_pipeline factory function"
```

---

### Task 9: Update Dependencies (pyproject.toml)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 更新 ingestion 依赖**

在 `pyproject.toml` 中，将 `ingestion` 依赖组：

```toml
ingestion = [
    "arq>=0.26",
    "PyMuPDF>=1.24",
    "python-docx>=0.8",
    "Pillow>=10.0",
    "markdown>=3.7",
]
```

替换为：

```toml
ingestion = [
    "arq>=0.26",                # 轻量异步任务队列（预留）
    "docling>=2.0",             # 多格式文档解析（PDF/Word/PPT/HTML/Markdown）
    "Pillow>=10.0",             # 图片处理（预留 MultiModalStage）
]
```

- [ ] **Step 2: 运行全部测试，验证兼容性**

```bash
python -m pytest tests/ -v
```
Expected: ALL PASS

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml
git commit -m "chore: update ingestion dependencies — replace PyMuPDF+python-docx with docling"
```

---

### Task 10: Integration Smoke Test

**Files:**
- Create: `tests/unit/ingestion/test_integration.py`

- [ ] **Step 1: 编写集成冒烟测试 — `tests/unit/ingestion/test_integration.py`**

```python
"""Ingestion Pipeline 集成冒烟测试"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestIntegrationSmoke:
    @pytest.mark.asyncio
    @patch("ingestion.models")
    @patch("ingestion.SentenceTransformer")
    async def test_markdown_file_full_pipeline(self, mock_st, mock_models):
        """从 Markdown 文件 → 分块 → embedding → FAISS 索引 的全链路冒烟测试"""
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            side_effect=lambda texts: [
                np.random.rand(1024).astype(np.float32) for _ in texts
            ]
        )
        mock_st.return_value = mock_model

        # Mock model manager
        mock_models.get_path.return_value = Path("/fake/models/bge")

        # Mock FAISS index_dir 到临时目录
        import tempfile
        import config.settings as _s
        original_dir = _s.settings.faiss.index_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            _s.settings.faiss.index_dir = tmpdir
            _s.settings.chunking.strategy = "fixed"  # 避免 SemanticChunker 调用 model

            try:
                from ingestion import create_default_pipeline

                pipeline = create_default_pipeline()

                # 创建临时 Markdown 文件
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, encoding="utf-8"
                ) as f:
                    f.write("# 测试文档\n\n这是第一段内容。\n\n## 第二节\n\n更多内容在这里。\n\n" * 20)
                    md_path = Path(f.name)

                try:
                    ctx = await pipeline.run(md_path, collection="test_integration")
                    assert ctx.status == "done"
                    assert len(ctx.chunks) > 0
                    assert ctx.chunks[0].embedding is not None
                    assert "parser_ms" in ctx.metadata
                    assert "chunker_ms" in ctx.metadata
                    assert "embedder_ms" in ctx.metadata

                    # 验证索引文件写入
                    idx_dir = Path(tmpdir) / "test_integration"
                    assert idx_dir.exists()
                    assert (idx_dir / "index.faiss").exists()
                    assert (idx_dir / "docstore.json").exists()
                finally:
                    md_path.unlink()
            finally:
                _s.settings.faiss.index_dir = original_dir
```

- [ ] **Step 2: 运行测试**

```bash
python -m pytest tests/unit/ingestion/test_integration.py -v
```
Expected: ALL PASS (1 test)

- [ ] **Step 3: 运行全部测试，确认无回归**

```bash
python -m pytest tests/ -v
```
Expected: ALL PASS

- [ ] **Step 4: 提交**

```bash
git add tests/unit/ingestion/test_integration.py
git commit -m "test(ingestion): add integration smoke test for full pipeline"
```

---

### Task 11: Add chunking config fields to defaults.yaml

**Files:**
- Modify: `config/defaults.yaml`

- [ ] **Step 1: 添加 SemanticChunker 配置项**

在 `config/defaults.yaml` 的 `chunking` 段中添加：

```yaml
chunking:
  chunk_size: 512
  overlap: 64
  strategy: semantic
  semantic_threshold_percentile: 0.9    # 新增
  semantic_buffer_size: 1               # 新增
```

同时更新 `src/config/settings.py` 中的 `ChunkingConfig`：

需要给 `ChunkingConfig` 添加两个新字段：
```python
semantic_threshold_percentile: float = 0.9
semantic_buffer_size: int = 1
```

- [ ] **Step 2: 验证解析正确**

```bash
python -c "from config import settings; print(settings.chunking.semantic_threshold_percentile)"
```
Expected: `0.9`

- [ ] **Step 3: 提交**

```bash
git add config/defaults.yaml src/config/settings.py
git commit -m "feat(config): add semantic_threshold_percentile and semantic_buffer_size to ChunkingConfig"
```

---

## Plan Self-Review

### 1. Spec Coverage

| Spec 章节 | 对应任务 | 状态 |
|-----------|---------|------|
| §2 架构概览（Stage/Pipeline/PipelineContext） | Task 1, 2, 3 | ✅ |
| §3 目录结构 | 所有 Task | ✅ |
| §4 ParserStage | Task 4 | ✅ |
| §5 ChunkerStage（3 种策略） | Task 5 | ✅ |
| §6 EmbedderStage | Task 6 | ✅ |
| §7 FAISSIndexWriter | Task 7 | ✅ |
| §8 编排 & 工厂函数 | Task 3, 8 | ✅ |
| §9 配置依赖 | Task 11 | ✅ |
| §10 未尽事项 | 未实现（按设计） | ✅ |

### 2. Placeholder Scan

- 无 TBD/TODO
- 所有步骤包含实际代码
- 所有测试包含完整断言
- 所有命令包含预期输出

### 3. Type Consistency

- `PipelineContext` 在各 Task 中一致引用 ✅
- `Stage` 协议签名 `run(ctx: PipelineContext) -> PipelineContext` 跨 Task 一致 ✅
- `splitter(text: str) -> list[Chunk]` 接口一致 ✅
- `embedding_model.encode(texts) -> list[ndarray]` 统一 ✅
- `IndexWriter.write(chunks, collection)` 签名一致 ✅
- `create_default_pipeline() -> IngestionPipeline` 返回类型一致 ✅



