# Model 模块统一推理接口实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `model` 模块新增统一本地模型推理接口（encode / rerank / generate），消除各业务模块分散加载模型的冗余代码。

**Architecture:** 新增 `src/model/inference.py` 承载模型缓存和推理核心逻辑；`ModelManager` 通过属性和方法委托暴露推理能力。完成后迁移 retrieval / ingestion 模块中的 `load_embedding_model()` / `load_cross_encoder()` 调用方。

**Tech Stack:** sentence-transformers, numpy, threading (stdlib)

**Spec:** `docs/superpowers/specs/2026-07-22-model-inference-api-design.md`

## Global Constraints

- 模型未下载时抛 RuntimeError，不自动触发下载
- 双检锁（DCL）保证模型单次加载
- encode/rerank 透传底层库 **kwargs
- generate() 抛 NotImplementedError，消息提及 llama-cpp-python + GGUF
- 遵循 TDD：先写测试再实现
- 遵循现有代码风格：TYPE_CHECKING 隔离、模块级 logger、中文 docstring
- **每步执行后必须验证（pytest），测试失败不可进入下一步**

---

### Task 1: 编写 `tests/unit/model/test_inference.py`（RED 阶段）

**Files:**
- Create: `tests/unit/model/test_inference.py`

**Interfaces:**
- Consumes: `model.models` 单例（已存在）、`model.inference` 模块（尚不存在，测试预期 FAIL）
- Produces: 9 个测试用例

- [ ] **Step 1: 创建测试文件**

```python
"""model 推理接口单元测试"""
import threading
import time

import numpy as np
import pytest

from model import models
from model import inference


# ============================================================================
# Fake 模型
# ============================================================================

class _FakeEncoder:
    """模拟 SentenceTransformer，统计实例化次数"""

    instances = 0
    _lock = threading.Lock()

    def __init__(self, path_or_model_name, device="cpu", **kwargs):
        time.sleep(0.05)
        with type(self)._lock:
            type(self).instances += 1
        self._path = str(path_or_model_name)
        self._device = device
        self._kwargs = kwargs

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)
        return np.array([[0.1, 0.2, 0.3]] * len(texts), dtype=np.float32)


class _FakeCrossEncoder:
    """模拟 CrossEncoder，统计实例化次数"""

    instances = 0
    _lock = threading.Lock()

    def __init__(self, path_or_model_name, device="cpu", **kwargs):
        with type(self)._lock:
            type(self).instances += 1
        self._path = str(path_or_model_name)
        self._device = device
        self._kwargs = kwargs

    def rank(self, query, documents, **kwargs):
        return [
            {"corpus_id": i, "score": 1.0 - i * 0.1}
            for i in range(len(documents))
        ]


# ============================================================================
# fixture
# ============================================================================

@pytest.fixture(autouse=True)
def reset_inference_cache(monkeypatch):
    """每个测试前重置推理缓存，确保测试隔离"""
    inference._reset_cache()
    monkeypatch.setattr(models, "get_path", lambda t: "/fake/model/path")
    monkeypatch.setattr(models, "_initialized", True, raising=False)
    yield
    inference._reset_cache()


# ============================================================================
# encode 测试
# ============================================================================

class TestEncode:
    def test_encode_single_text(self, monkeypatch):
        """单条文本返回 1D ndarray"""
        monkeypatch.setattr(inference, "SentenceTransformer", _FakeEncoder)
        result = models.encode("你好世界")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_encode_multiple_texts(self, monkeypatch):
        """多条文本返回 2D ndarray"""
        monkeypatch.setattr(inference, "SentenceTransformer", _FakeEncoder)
        texts = ["文本一", "文本二", "文本三"]
        result = models.encode(texts)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (3, 3)

    def test_encode_kwargs_passthrough(self, monkeypatch):
        """**kwargs 透传给底层 model.encode()"""
        received_kwargs = {}

        class _KwargsRecorder:
            def encode(self, texts, **kwargs):
                received_kwargs.update(kwargs)
                n = len(texts) if isinstance(texts, list) else 1
                return np.array([[0.1]] * n)

        monkeypatch.setattr(inference, "SentenceTransformer", _KwargsRecorder)
        models.encode(["文本"], batch_size=16, normalize_embeddings=True)
        assert received_kwargs.get("batch_size") == 16
        assert received_kwargs.get("normalize_embeddings") is True

    def test_encode_model_not_downloaded_raises(self, monkeypatch):
        """模型未下载时抛 RuntimeError"""
        monkeypatch.setattr(models, "get_path", lambda t: None)
        with pytest.raises(RuntimeError, match="模型未下载|Embedding 模型未下载"):
            models.encode("测试")


# ============================================================================
# rerank 测试
# ============================================================================

class TestRerank:
    def test_rerank_returns_ranked_list(self, monkeypatch):
        """返回 list[dict]，每项含 corpus_id + score"""
        monkeypatch.setattr(inference, "CrossEncoder", _FakeCrossEncoder)
        result = models.rerank("查询", ["文档A", "文档B", "文档C"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert "corpus_id" in result[0]
        assert "score" in result[0]
        assert result[0]["score"] > result[-1]["score"]

    def test_rerank_kwargs_passthrough(self, monkeypatch):
        """**kwargs 透传给底层 model.rank()"""
        received_kwargs = {}

        class _KwargsRecorder:
            def rank(self, query, documents, **kwargs):
                received_kwargs.update(kwargs)
                return [{"corpus_id": 0, "score": 1.0}]

        monkeypatch.setattr(inference, "CrossEncoder", _KwargsRecorder)
        models.rerank("查询", ["文档"], top_k=10, return_documents=True)
        assert received_kwargs.get("top_k") == 10
        assert received_kwargs.get("return_documents") is True

    def test_rerank_model_not_downloaded_raises(self, monkeypatch):
        """模型未下载时抛 RuntimeError"""
        monkeypatch.setattr(models, "get_path", lambda t: None)
        with pytest.raises(RuntimeError, match="模型未下载|Rerank 模型未下载"):
            models.rerank("查询", ["文档"])


# ============================================================================
# generate 测试
# ============================================================================

class TestGenerate:
    def test_generate_raises_not_implemented(self):
        """generate() 抛 NotImplementedError，消息含方案说明"""
        with pytest.raises(NotImplementedError) as exc_info:
            models.generate("什么是最佳方案？")
        msg = str(exc_info.value)
        assert "llama-cpp-python" in msg
        assert "GGUF" in msg


# ============================================================================
# 缓存 & 线程安全测试
# ============================================================================

class TestModelCaching:
    def test_model_instance_cached(self, monkeypatch):
        """连续两次访问返回同一实例"""
        _FakeEncoder.instances = 0
        monkeypatch.setattr(inference, "SentenceTransformer", _FakeEncoder)

        m1 = models.embedding_model
        m2 = models.embedding_model

        assert m1 is m2
        assert _FakeEncoder.instances == 1

    def test_cross_encoder_instance_cached(self, monkeypatch):
        """CrossEncoder 连续访问也复用"""
        _FakeCrossEncoder.instances = 0
        monkeypatch.setattr(inference, "CrossEncoder", _FakeCrossEncoder)

        ce1 = models.cross_encoder
        ce2 = models.cross_encoder

        assert ce1 is ce2
        assert _FakeCrossEncoder.instances == 1

    def test_concurrent_load_once(self, monkeypatch):
        """4 线程并发首次访问只加载一次"""
        _FakeEncoder.instances = 0
        monkeypatch.setattr(inference, "SentenceTransformer", _FakeEncoder)

        errors: list = []

        def load_model():
            try:
                _ = models.embedding_model
            except Exception as e:
                errors.append(repr(e))

        threads = [threading.Thread(target=load_model) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert _FakeEncoder.instances == 1, (
            f"模型应只加载一次，实际加载 {_FakeEncoder.instances} 次"
        )
```

- [ ] **Step 2: 运行测试，确认全部 FAIL**

```bash
pytest tests/unit/model/test_inference.py -v
```

预期：全部 9 个测试 FAIL（`inference` 模块和 `models.encode/rerank/generate` 尚不存在）

- [ ] **Step 3: 提交**

```bash
git add tests/unit/model/test_inference.py
git commit -m "test: model 推理接口测试（RED — 尚未实现）"
```

---

### Task 2: 实现 `src/model/inference.py`（GREEN 阶段）

**Files:**
- Create: `src/model/inference.py`

**Interfaces:**
- Produces:
  - `_get_embedding_model() -> SentenceTransformer`
  - `_get_cross_encoder() -> CrossEncoder`
  - `encode(texts: str | list[str], **kwargs) -> np.ndarray`
  - `rerank(query: str, documents: list[str], **kwargs) -> list[dict]`
  - `generate(prompt: str, **kwargs) -> str`
  - `_reset_cache() -> None`

- [ ] **Step 1: 创建 `inference.py`**

```python
"""
模型推理引擎 — 统一加载 + 调用本地 Embedding / Rerank / LLM 模型

进程级单例缓存（双检锁），线程安全。
"""
import threading

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ============================================================================
# 模块级缓存
# ============================================================================

_embedding_model: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None
_embedding_lock = threading.Lock()
_cross_encoder_lock = threading.Lock()

_GENERATE_NOT_IMPLEMENTED_MSG = (
    "generate() 尚未实现。推荐方案：llama-cpp-python + GGUF 量化模型。\n"
    "  - CPU 友好，内存占用低（INT4 量化后约 4-8 GB）\n"
    "  - 安装: pip install llama-cpp-python\n"
    "  - 使用: from llama_cpp import Llama; llm = Llama(model_path=\"model.gguf\")\n"
    "  - 项目当前 LLM 生成走云端 DeepSeek API，本地推理作为后续迭代方向。"
)


# ============================================================================
# 模型加载（内部）
# ============================================================================

def _get_embedding_model() -> SentenceTransformer:
    """获取 SentenceTransformer 实例（懒加载 + 双检锁）"""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_lock:
        if _embedding_model is None:
            from config import settings
            from model import models

            path = models.get_path("embedding")
            if path is None:
                raise RuntimeError(
                    "Embedding 模型未下载，请先执行 "
                    "`from model import models; models.download('embedding')`"
                )
            _embedding_model = SentenceTransformer(
                str(path), device=settings.embedding.device
            )
    return _embedding_model


def _get_cross_encoder() -> CrossEncoder:
    """获取 CrossEncoder 实例（懒加载 + 双检锁）"""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    with _cross_encoder_lock:
        if _cross_encoder is None:
            from config import settings
            from model import models

            path = models.get_path("rerank")
            if path is None:
                raise RuntimeError(
                    "Rerank 模型未下载，请先执行 "
                    "`from model import models; models.download('rerank')`"
                )
            _cross_encoder = CrossEncoder(
                str(path), device=settings.embedding.device
            )
    return _cross_encoder


# ============================================================================
# 公共推理接口
# ============================================================================

def encode(texts: str | list[str], **kwargs) -> np.ndarray:
    """对文本进行 embedding 编码。

    Args:
        texts: 单条文本或文本列表
        **kwargs: 透传给 SentenceTransformer.encode()

    Returns:
        np.ndarray — 单条返回 1D，多条返回 2D
    """
    model = _get_embedding_model()
    return model.encode(texts, **kwargs)


def rerank(query: str, documents: list[str], **kwargs) -> list[dict]:
    """对查询与候选文档进行相关性排序。

    Args:
        query: 查询文本
        documents: 候选文档文本列表
        **kwargs: 透传给 CrossEncoder.rank()

    Returns:
        list[dict] — [{"corpus_id": int, "score": float}, ...]
    """
    model = _get_cross_encoder()
    return model.rank(query, documents, **kwargs)


def generate(prompt: str, **kwargs) -> str:
    """LLM 文本生成（预留接口，当前未实现）。

    推荐方案：llama-cpp-python + GGUF 量化模型。
    项目当前 LLM 生成走云端 DeepSeek API，本地推理作为后续迭代方向。

    Raises:
        NotImplementedError: 始终抛出，消息体包含方案说明。
    """
    raise NotImplementedError(_GENERATE_NOT_IMPLEMENTED_MSG)


# ============================================================================
# 测试辅助
# ============================================================================

def _reset_cache() -> None:
    """重置模块级模型缓存（仅用于测试隔离）"""
    global _embedding_model, _cross_encoder
    _embedding_model = None
    _cross_encoder = None
```

- [ ] **Step 2: 运行测试，确认全部 PASS**

```bash
pytest tests/unit/model/test_inference.py -v
```

预期：9 passed

- [ ] **Step 3: 提交**

```bash
git add src/model/inference.py
git commit -m "feat: model 推理引擎 — encode/rerank/generate 统一接口"
```

---

### Task 3: 添加委托方法到 `ModelManager`

**Files:**
- Modify: `src/model/manager.py` — 添加推理属性和方法
- Modify: `src/model/__init__.py` — 添加 `inference` 模块导出

**Interfaces:**
- Produces:
  - `ModelManager.embedding_model` property → `SentenceTransformer`
  - `ModelManager.cross_encoder` property → `CrossEncoder`
  - `ModelManager.encode(texts, **kwargs)` → `np.ndarray`
  - `ModelManager.rerank(query, docs, **kwargs)` → `list[dict]`
  - `ModelManager.generate(prompt, **kwargs)` → `str`

- [ ] **Step 1: 更新 `manager.py` 顶部的 TYPE_CHECKING 块**

在第 10 行的 TYPE_CHECKING 块中，于 `FinetuneConfig` 导入之后添加：

```python
    import numpy as np
    from sentence_transformers import CrossEncoder, SentenceTransformer
```

- [ ] **Step 2: 在 `manager.py` 中插入推理 API 段落**

在第 176 行（`# 微调 API` 注释之前）插入：

```python
    # ========================================================================
    # 推理 API
    # ========================================================================

    @property
    def embedding_model(self) -> "SentenceTransformer":
        """获取 Embedding 模型实例（懒加载 + 双检锁，进程级缓存）"""
        self._ensure_init()
        from .inference import _get_embedding_model
        return _get_embedding_model()

    @property
    def cross_encoder(self) -> "CrossEncoder":
        """获取 CrossEncoder 实例（懒加载 + 双检锁，进程级缓存）"""
        self._ensure_init()
        from .inference import _get_cross_encoder
        return _get_cross_encoder()

    def encode(self, texts: str | list[str], **kwargs) -> "np.ndarray":
        """对文本进行 embedding 编码（透传 SentenceTransformer.encode()）"""
        self._ensure_init()
        from .inference import encode as _encode
        return _encode(texts, **kwargs)

    def rerank(self, query: str, documents: list[str], **kwargs) -> list[dict]:
        """对查询与候选文档进行相关性排序（透传 CrossEncoder.rank()）"""
        self._ensure_init()
        from .inference import rerank as _rerank
        return _rerank(query, documents, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """LLM 文本生成（预留接口，当前未实现）

        Raises:
            NotImplementedError: 始终抛出，消息体包含 llama-cpp-python + GGUF 方案说明。
        """
        self._ensure_init()
        from .inference import generate as _generate
        return _generate(prompt, **kwargs)
```

- [ ] **Step 3: 更新 `__init__.py`**

在现有 `from .manager import ModelManager, models` 之后添加：

```python
from . import inference
```

在 `__all__` 列表中添加 `"inference"`：

```python
__all__ = [
    "AutoStrategy",
    "DownloadStrategy",
    "HfStrategy",
    "ModelDownloader",
    "ModelManager",
    "MsStrategy",
    "__version__",
    "inference",
    "models",
]
```

- [ ] **Step 4: 运行全量 model 测试确认无回归**

```bash
pytest tests/unit/model/ -v
```

预期：全部 PASS（含 test_inference.py 的 9 个测试）

- [ ] **Step 5: 提交**

```bash
git add src/model/manager.py src/model/__init__.py
git commit -m "feat: ModelManager 添加 encode/rerank/generate 推理接口"
```

---

### Task 4: 迁移 `retrieval/layer.py` — 先更新调用方

**Files:**
- Modify: `src/retrieval/layer.py`

**Interfaces:**
- Consumes: `model.models`

> **注意：本任务必须在 Task 5/6（删除旧函数）之前执行，否则中间状态编译失败。**

- [ ] **Step 1: 更新 import 和懒加载调用**

**Line 16**: 在 `from models.enums import RetrievalEval` 之后添加：
```python
from model import models
```

**Line 23**: 将
```python
from retrieval.reranker import Reranker, load_cross_encoder, mmr_select
```
改为：
```python
from retrieval.reranker import Reranker, mmr_select
```

**Line 25**: 将
```python
from retrieval.vector_retriever import VectorRetriever, load_embedding_model
```
改为：
```python
from retrieval.vector_retriever import VectorRetriever
```

**Line 49**: 将
```python
                self._encoder = load_embedding_model()
```
改为：
```python
                self._encoder = models.embedding_model
```

**Line 57**: 将
```python
                self._cross_encoder = load_cross_encoder()
```
改为：
```python
                self._cross_encoder = models.cross_encoder
```

- [ ] **Step 2: 运行 retrieval 测试确认无回归**

```bash
pytest tests/unit/retrieval/ -v
```

- [ ] **Step 3: 提交**

```bash
git add src/retrieval/layer.py
git commit -m "refactor: retrieval/layer 使用 models.embedding_model / models.cross_encoder"
```

---

### Task 5: 清理 `retrieval/vector_retriever.py`

**Files:**
- Modify: `src/retrieval/vector_retriever.py`

**前置条件：** Task 4 已完成

- [ ] **Step 1: 删除 `load_embedding_model()` 及关联代码**

删除：
- `import threading`
- `_embedding_model` 全局变量
- `_model_lock`
- `load_embedding_model()` 函数

TYPE_CHECKING 中保留 `SentenceTransformer`（`VectorRetriever.__init__` 类型标注需要）。

修改后的文件：

```python
"""VectorRetriever — 查询编码 + FAISS 向量召回"""
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from retrieval.store import FAISSStore


class VectorRetriever:
    """查询 → encoder 编码（COSINE 时归一化，与写入侧一致）→ FAISS 搜索"""

    def __init__(self, store: "FAISSStore", encoder: "SentenceTransformer"):
        self._store = store
        self._encoder = encoder

    def retrieve(self, query: str, k: int) -> list[str]:
        from config import settings

        vec = np.asarray(self._encoder.encode([query]), dtype=np.float32)
        if settings.faiss.metric_type == "COSINE":
            faiss.normalize_L2(vec)
        return self._store.search(vec[0], k)
```

- [ ] **Step 2: 运行 retrieval 测试确认无回归**

```bash
pytest tests/unit/retrieval/ -v
```

- [ ] **Step 3: 提交**

```bash
git add src/retrieval/vector_retriever.py
git commit -m "refactor: 移除 vector_retriever.load_embedding_model"
```

---

### Task 6: 清理 `retrieval/reranker.py`

**Files:**
- Modify: `src/retrieval/reranker.py`

**前置条件：** Task 4 已完成

- [ ] **Step 1: 删除 `load_cross_encoder()` 及关联代码**

删除：`import threading`、`_cross_encoder`、`_ce_lock`、`load_cross_encoder()`。

TYPE_CHECKING 中保留 `CrossEncoder`（`Reranker.__init__` 需要）。

`Reranker` 类和 `mmr_select` / `_normalize` 函数保持不变。

- [ ] **Step 2: 运行 retrieval 测试确认无回归**

```bash
pytest tests/unit/retrieval/ -v
```

- [ ] **Step 3: 提交**

```bash
git add src/retrieval/reranker.py
git commit -m "refactor: 移除 reranker.load_cross_encoder"
```

---

### Task 7: 迁移 `ingestion/__init__.py`

**Files:**
- Modify: `src/ingestion/__init__.py`

- [ ] **Step 1: 替换手动模型加载为 `models.embedding_model`**

删除模块级变量 `_cached_embedding_model`、`_model_load_lock` 和 `import threading`。

删除手动加载逻辑（`if _cached_embedding_model is None: ...`），改为直接使用 `models.embedding_model`。

修改后的文件：

```python
"""Ingestion 模块 — 离线文档处理 Pipeline"""

from model import models

from .chunker import ChunkerStage
from .embedder import EmbedderStage
from .indexer import FAISSIndexWriter
from .parser import ParserStage
from .pipeline import IngestionPipeline


def create_default_pipeline() -> IngestionPipeline:
    """组装默认的 ingestion pipeline

    Chunker（SemanticChunker）和 Embedder 共享 models.embedding_model 实例。
    """
    embedding_model = models.embedding_model

    return IngestionPipeline(
        stages=[
            ParserStage(),
            ChunkerStage(embedding_model=embedding_model),
            EmbedderStage(embedding_model=embedding_model),
        ],
        index_writer=FAISSIndexWriter(),
    )
```

- [ ] **Step 2: 运行 ingestion 测试确认无回归**

```bash
pytest tests/unit/ingestion/ -v
```

- [ ] **Step 3: 提交**

```bash
git add src/ingestion/__init__.py
git commit -m "refactor: ingestion 使用 models.embedding_model 替代手动加载"
```

---

### Task 8: 更新 `tests/unit/ingestion/test_init.py`

**Files:**
- Modify: `tests/unit/ingestion/test_init.py`

- [ ] **Step 1: 重写测试，通过 `models` 验证**

```python
"""create_default_pipeline 模型加载线程安全测试（通过 models 推理接口验证）"""
import threading
import time

from model import models
from model import inference


class _SlowFakeModel:
    """模拟加载耗时的 SentenceTransformer"""

    instances = 0
    _count_lock = threading.Lock()

    def __init__(self, path_or_name, device="cpu", **kwargs):
        time.sleep(0.05)
        with type(self)._count_lock:
            type(self).instances += 1


class TestPipelineModelLoading:
    def test_concurrent_load_loads_model_once(self, monkeypatch):
        """4 线程并发创建 pipeline，模型只加载一次"""
        inference._reset_cache()
        monkeypatch.setattr(models, "get_path", lambda t: "/fake/path")
        monkeypatch.setattr(models, "_initialized", True)
        _SlowFakeModel.instances = 0
        monkeypatch.setattr(
            inference, "SentenceTransformer", _SlowFakeModel
        )

        import ingestion

        errors: list = []

        def build():
            try:
                ingestion.create_default_pipeline()
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=build) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert _SlowFakeModel.instances == 1, (
            f"GB 级模型应只加载一次，实际加载 {_SlowFakeModel.instances} 次"
        )
```

- [ ] **Step 2: 运行该测试确认 PASS**

```bash
pytest tests/unit/ingestion/test_init.py -v
```

- [ ] **Step 3: 提交**

```bash
git add tests/unit/ingestion/test_init.py
git commit -m "test: 更新 ingestion 线程安全测试，通过 models.inference 验证"
```

---

### Task 9: 全量回归验证

**Files:** 无新建/修改

- [ ] **Step 1: 运行全量单元测试**

```bash
pytest tests/unit/ -v
```

预期：全部 PASS

- [ ] **Step 2: 确认无残留引用**

```bash
grep -r "load_embedding_model\|load_cross_encoder" src/ tests/
```

预期：无输出

- [ ] **Step 3: 提交（如有遗漏修复）**

```bash
git add -A && git commit -m "chore: 全量测试回归通过，清理残留引用"
```
