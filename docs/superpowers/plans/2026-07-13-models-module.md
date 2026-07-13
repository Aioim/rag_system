# Models 模块 — 实施计划

> **For agentic workers:** 使用 TDD 逐任务实现

**Goal:** 创建共享数据模型层（Document/Chunk/Session/API models/Enums），从 ingestion 提取已有定义并补充在线 Pipeline 所需模型。

**Architecture:** 纯 dataclass 层，无外部依赖。ingestion/context.py 改为从 models 重导出。

## Global Constraints

- Python >= 3.11, ruff line-length 100, double quotes
- 测试目录 `tests/unit/models/`，源目录 `src/models/`
- 纯 dataclass，不引入 pydantic（API 层除外）
- TDD: 先写测试→验证失败→实现→验证通过→提交

## File Map

| 文件 | 职责 |
|------|------|
| `src/models/enums.py` | Intent, RetrievalEval, DocumentStatus |
| `src/models/chunk.py` | Chunk（从 ingestion 提取 + rerank_score） |
| `src/models/document.py` | Document（合并 ingestion + spec） |
| `src/models/session.py` | Session + Message |
| `src/models/api.py` | ChatRequest/Response, SearchRequest/Response, Source |
| `src/models/context.py` | PipelineContext（在线版） |
| `src/models/__init__.py` | 统一导出 |
| `src/ingestion/context.py` | 改为从 models 重导出 |

---

### Task 1: Enums (enums.py)

**Files:**
- Create: `tests/unit/models/__init__.py`
- Create: `tests/unit/models/test_enums.py`
- Create: `src/models/__init__.py`（空）
- Create: `src/models/enums.py`

**Interfaces:**
- `Intent`: CONCEPT / PROCEDURE / COMPARE / LOOKUP
- `RetrievalEval`: SUFFICIENT / NEED_MORE / INSUFFICIENT
- `DocumentStatus`: PENDING / PARSING / CHUNKING / EMBEDDING / DONE / FAILED

- [ ] **Step 1: 测试**

```python
"""Enums 测试"""
from models.enums import Intent, RetrievalEval, DocumentStatus


class TestIntent:
    def test_values(self):
        assert Intent.CONCEPT.value == "concept"
        assert Intent.PROCEDURE.value == "procedure"
        assert Intent.COMPARE.value == "compare"
        assert Intent.LOOKUP.value == "lookup"

    def test_from_string(self):
        assert Intent("concept") == Intent.CONCEPT


class TestRetrievalEval:
    def test_values(self):
        assert RetrievalEval.SUFFICIENT.value == "sufficient"
        assert RetrievalEval.NEED_MORE.value == "need_more"
        assert RetrievalEval.INSUFFICIENT.value == "insufficient"


class TestDocumentStatus:
    def test_values(self):
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.DONE.value == "done"
        assert DocumentStatus.FAILED.value == "failed"
```

- [ ] **Step 2: 实现**

```python
"""共享枚举类型"""
from enum import Enum


class Intent(str, Enum):
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    COMPARE = "compare"
    LOOKUP = "lookup"


class RetrievalEval(str, Enum):
    SUFFICIENT = "sufficient"
    NEED_MORE = "need_more"
    INSUFFICIENT = "insufficient"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    DONE = "done"
    FAILED = "failed"
```

- [ ] **Step 3: 验证 → 提交**

```bash
python -m pytest tests/unit/models/test_enums.py -v
git add tests/unit/models/ src/models/ && git commit -m "feat(models): add enums — Intent, RetrievalEval, DocumentStatus"
```

---

### Task 2: Chunk (chunk.py)

**Files:** Create `tests/unit/models/test_chunk.py`, `src/models/chunk.py`

**From ingestion add `rerank_score`.**

- [ ] **Step 1: 测试**

```python
"""Chunk 测试"""
from models.chunk import Chunk


class TestChunk:
    def test_minimal(self):
        c = Chunk(chunk_id="c1", doc_id="d1", text="hello", chunk_index=0)
        assert c.rerank_score == 0.0
        assert c.embedding is None

    def test_linked_list(self):
        c = Chunk(chunk_id="c-mid", doc_id="d1", text="mid", chunk_index=1,
                   prev_chunk_id="c-0", next_chunk_id="c-2")
        assert c.prev_chunk_id == "c-0"
        assert c.next_chunk_id == "c-2"

    def test_rerank_score(self):
        c = Chunk(chunk_id="c1", doc_id="d1", text="x", chunk_index=0,
                   rerank_score=0.85)
        assert c.rerank_score == 0.85
```

- [ ] **Step 2: 实现** — 从 `ingestion/context.py` 复制 Chunk 定义，增加 `rerank_score: float = 0.0`

- [ ] **Step 3: 验证 → 提交**

---

### Task 3: Document (document.py)

**Files:** Create `tests/unit/models/test_document.py`, `src/models/document.py`

**合并两个版本：ingestion 的 source_path/raw_text + spec 的 status/created_at。**

- [ ] **Step 1: 测试**

```python
"""Document 测试"""
from datetime import datetime
from pathlib import Path
from models.document import Document
from models.enums import DocumentStatus


class TestDocument:
    def test_minimal(self):
        doc = Document(doc_id="d1", source_path=Path("test.pdf"), file_type="pdf")
        assert doc.status == DocumentStatus.PENDING
        assert doc.raw_text == ""
        assert doc.created_at is not None

    def test_full(self):
        doc = Document(
            doc_id="d2", source_path=Path("r.docx"), file_type="docx",
            title="报告", raw_text="# 内容", collection="tech",
            status=DocumentStatus.DONE, metadata={"pages": 10},
        )
        assert doc.title == "报告"
        assert doc.collection == "tech"
        assert doc.status == DocumentStatus.DONE
```

- [ ] **Step 2: 实现**

```python
@dataclass
class Document:
    doc_id: str
    source_path: Path
    file_type: str
    title: str = ""
    raw_text: str = ""
    collection: str = "default"
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 3: 验证 → 提交**

---

### Task 4: Session (session.py)

**Files:** Create `tests/unit/models/test_session.py`, `src/models/session.py`

- [ ] **Step 1: 测试**

```python
"""Session 测试"""
from datetime import datetime
from models.session import Session, Message


class TestMessage:
    def test_construction(self):
        msg = Message(role="user", content="什么是RAG？")
        assert msg.role == "user"
        assert msg.timestamp is not None


class TestSession:
    def test_minimal(self):
        s = Session(session_id="s1")
        assert s.messages == []
        assert s.context_summary is None

    def test_add_message(self):
        s = Session(session_id="s1")
        s.messages.append(Message(role="user", content="hello"))
        s.messages.append(Message(role="assistant", content="hi"))
        assert len(s.messages) == 2
```

- [ ] **Step 2: 实现**

```python
@dataclass
class Message:
    role: str                           # user / assistant / system
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    context_summary: str | None = None
    current_topic: str | None = None
    topic_embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 3: 验证 → 提交**

---

### Task 5: API Models (api.py)

**Files:** Create `tests/unit/models/test_api.py`, `src/models/api.py`

- [ ] **实现**

```python
@dataclass
class Source:
    doc_id: str
    doc_title: str
    chunk_text: str
    score: float


@dataclass
class ChatRequest:
    query: str
    session_id: str | None = None
    collection: str = "default"
    stream: bool = False
    top_k: int = 5


@dataclass
class ChatResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    session_id: str = ""
    confidence: float = 0.0
    is_fallback: bool = False


@dataclass
class SearchRequest:
    query: str
    collection: str = "default"
    top_k: int = 10


@dataclass
class SearchResponse:
    results: list[Source] = field(default_factory=list)
    search_type: str = "hybrid"
```

---

### Task 6: PipelineContext (context.py)

**Files:** Create `tests/unit/models/test_context.py`, `src/models/context.py`

**在线版 PipelineContext（与 ingestion 版不同，用于 QA 链路）。**

```python
@dataclass
class PipelineContext:
    query: str
    rewritten_queries: list[str] = field(default_factory=list)
    intent: Intent | None = None
    collection: str = "default"
    candidates: list[Chunk] = field(default_factory=list)
    reranked: list[Chunk] = field(default_factory=list)
    session: Session | None = None
    assembled_prompt: str = ""
    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_eval: RetrievalEval | None = None
    fallback_level: str = ""
    is_fallback: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None
    metadata: dict = field(default_factory=dict)
```

---

### Task 7: __init__.py 导出

```python
from models.enums import Intent, RetrievalEval, DocumentStatus
from models.chunk import Chunk
from models.document import Document
from models.session import Session, Message
from models.api import ChatRequest, ChatResponse, SearchRequest, SearchResponse, Source
from models.context import PipelineContext
```

---

### Task 8: 更新 ingestion/context.py

改为 `from models import Document, Chunk, PipelineContext as IngestionContext` 重导出，保持向后兼容。注意：ingestion 的 PipelineContext 与 online 的 PipelineContext 结构不同，需要重命名区分。

方案：ingestion/context.py 保留自己的 `PipelineContext` 类名不变，`Document` 和 `Chunk` 改为从 models 导入。StageError 也保留在 ingestion。

---

## 执行顺序

```
Task 1 (enums) → Task 2 (chunk) → Task 3 (document)
    → Task 4 (session) → Task 5 (api) → Task 6 (context)
    → Task 7 (__init__) → Task 8 (ingestion compat)
```

Task 2-5 可并行（都只依赖 enums）。Task 8 放最后确保不破坏现有 ingestion 测试。
