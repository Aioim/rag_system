# Retrieval 模块实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现在线 Pipeline 的检索层：两路召回（向量+BM25）→ RRF 融合 → 上下文扩展 → CrossEncoder 精排 + MMR → Self-RAG 自评。

**Architecture:** 自建组件式 Pipeline（仿 query 模块模式）：`RetrievalLayer` 主编排器 + 单职责组件（store / vector_retriever / bm25_retriever / fusion / expander / reranker / evaluator），`get_retrieval_layer()` 双重检查锁单例。数据源为 ingestion 已持久化的 `data/faiss_indexes/{collection}/index.faiss` + `docstore.json`（只读），BM25 索引启动时从 docstore 在内存构建。

**Tech Stack:** Python 3.11 / faiss-cpu / sentence-transformers（SentenceTransformer + CrossEncoder）/ jieba + rank-bm25 / pytest + pytest-asyncio（asyncio_mode=auto）

**设计文档:** `docs/superpowers/specs/2026-07-15-retrieval-design.md`

## Global Constraints

- 运行环境 Windows；测试从仓库根目录执行，`pyproject.toml` 已配 `pythonpath = ["src"]`，源码内以顶层包名导入（`from retrieval.store import ...`，不带 `src.` 前缀）
- 不改动 ingestion 模块的任何代码；对 FAISS 索引与 docstore 只读
- 模型只经 `models.get_path(type)` 获取本地路径（返回 `Path | None`），未下载时抛错提示，不自动下载
- 日志统一 `from logger import logger`，printf 风格参数（`logger.warning("... %s", x)`）
- 配置统一 `from config import settings` 且在函数内部导入（延迟初始化，见 ingestion/indexer.py 先例）
- 同步重活（模型推理 / faiss IO / BM25 构建）一律 `loop.run_in_executor(None, ...)`，不阻塞事件循环
- 中文 docstring；每个新测试目录需含 `__init__.py`
- commit message 遵循仓库惯例：`feat(retrieval): ...` / `test(retrieval): ...`

---

### Task 1: 依赖与配置扩展

**Files:**
- Modify: `pyproject.toml`（`[project.optional-dependencies]` 的 `retrieval` 组）
- Modify: `src/config/settings.py:42-51`（`RetrievalConfig`）
- Modify: `config/defaults.yaml:46-54`（`retrieval:` 节）
- Test: `tests/unit/config/__init__.py`（新建空包）、`tests/unit/config/test_retrieval_config.py`

**Interfaces:**
- Produces: `settings.retrieval.max_rerank_candidates: int = 30`（后续 Task 9 使用）

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/config/__init__.py`（空文件）和 `tests/unit/config/test_retrieval_config.py`：

```python
"""RetrievalConfig 新增字段测试"""
from config import settings


def test_max_rerank_candidates_default():
    assert settings.retrieval.max_rerank_candidates == 30
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/config/test_retrieval_config.py -v`
Expected: FAIL，`AttributeError: ... max_rerank_candidates`

- [ ] **Step 3: 实现**

`src/config/settings.py` 的 `RetrievalConfig`（42-51 行）中 `rrf_k: int = 60` 之后插入一行：

```python
    max_rerank_candidates: int = 30
```

`config/defaults.yaml` 的 `retrieval:` 节中 `rrf_k: 60` 行之后插入：

```yaml
  max_rerank_candidates: 30      # RRF 后进入精排的候选上限（CPU CrossEncoder 延迟保护）
```

`pyproject.toml` 的 `retrieval` 可选依赖组末尾追加：

```toml
    "jieba>=0.42",
```

`pyproject.toml` 的 `[tool.ruff.lint.isort]` 中 `known-first-party` 列表追加 `"retrieval"`（否则新模块 import 会被 ruff 归为第三方组导致 I001 误报）：

```toml
known-first-party = ["config", "logger", "security", "model", "models", "ingestion", "retrieval"]
```

- [ ] **Step 4: 安装缺失依赖**

Run: `pip install "jieba>=0.42" "rank-bm25>=0.2"`
Expected: Successfully installed jieba-x.x rank-bm25-x.x
验证: `python -c "import jieba, rank_bm25; print('OK')"` → `OK`

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/unit/config/test_retrieval_config.py -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/config/settings.py config/defaults.yaml tests/unit/config/
git commit -m "feat(retrieval): add max_rerank_candidates config and jieba dependency"
```

---

### Task 2: FAISSStore — 索引与 docstore 只读访问

**Files:**
- Create: `src/retrieval/__init__.py`（本任务先建**空文件**占位，Task 9 填充单例工厂）
- Create: `src/retrieval/store.py`
- Test: `tests/unit/retrieval/__init__.py`（空）、`tests/unit/retrieval/conftest.py`、`tests/unit/retrieval/test_store.py`

**Interfaces:**
- Consumes: `ingestion.indexer.FAISSIndexWriter`（仅测试中用于造索引）、`models.chunk.Chunk`
- Produces:
  - `class FAISSStore`: 属性 `collection: str`、`version: int`（reload 递增）、`is_empty: bool`；方法 `load() -> None`（幂等；collection 目录缺失抛 `ValueError`）、`get_chunk(chunk_id: str) -> Chunk | None`（每次返回新 Chunk 实例）、`search(vector: np.ndarray, k: int) -> list[str]`（一维 float32 向量入参，返回按相关度降序的 chunk_id）、`reconstruct(chunk_id: str) -> np.ndarray | None`、`all_chunks() -> list[tuple[str, str]]`（(chunk_id, text)）、`reload() -> None`
  - `get_store(collection: str) -> FAISSStore`（模块级缓存 + 触发 load）
  - `reset_stores() -> None`（测试用）

- [ ] **Step 1: 写共享 conftest**

`tests/unit/retrieval/__init__.py`（空文件）。`tests/unit/retrieval/conftest.py`：

```python
"""retrieval 测试共享 fixture — 临时小维度 FAISS 环境"""
import pytest

from config import settings
from ingestion.context import Chunk
from ingestion.indexer import FAISSIndexWriter

DIM = 8


def one_hot(i: int, scale: float = 1.0) -> list[float]:
    v = [0.0] * DIM
    v[i] = scale
    return v


def make_chunk(
    i: int,
    text: str,
    vec: list[float],
    doc_id: str = "d1",
    prev_id: str | None = None,
    next_id: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        doc_id=doc_id,
        text=text,
        chunk_index=i,
        prev_chunk_id=prev_id,
        next_chunk_id=next_id,
        embedding=vec,
    )


def write_chunks(chunks: list[Chunk], collection: str = "test") -> None:
    FAISSIndexWriter().write(chunks, collection)


@pytest.fixture
def faiss_env(tmp_path):
    """tmp 索引目录 + 8 维 + FLAT 索引；结束后还原配置并清空 store 缓存"""
    cfg = settings.faiss
    saved = (cfg.index_dir, cfg.dimension, cfg.index_type, cfg.nlist)
    cfg.index_dir = tmp_path
    cfg.dimension = DIM
    cfg.index_type = "FLAT"
    cfg.nlist = 4
    from retrieval.store import reset_stores

    reset_stores()
    yield tmp_path
    cfg.index_dir, cfg.dimension, cfg.index_type, cfg.nlist = saved
    reset_stores()
```

- [ ] **Step 2: 写失败测试**

`tests/unit/retrieval/test_store.py`：

```python
"""FAISSStore 测试"""
import numpy as np
import pytest

from config import settings
from retrieval.store import FAISSStore, get_store, reset_stores
from tests.unit.retrieval.conftest import make_chunk, one_hot, write_chunks


class TestFAISSStore:
    def test_missing_collection_raises(self, faiss_env):
        with pytest.raises(ValueError, match="nonexistent"):
            get_store("nonexistent")

    def test_get_chunk_builds_chunk(self, faiss_env):
        write_chunks([
            make_chunk(0, "文本零", one_hot(0), prev_id=None, next_id="c1"),
            make_chunk(1, "文本一", one_hot(1), prev_id="c0"),
        ])
        store = get_store("test")
        c = store.get_chunk("c0")
        assert c is not None
        assert c.text == "文本零"
        assert c.doc_id == "d1"
        assert c.next_chunk_id == "c1"
        assert store.get_chunk("missing") is None
        # 每次返回新实例（防调用方污染 docstore）
        assert store.get_chunk("c0") is not c

    def test_search_returns_ranked_chunk_ids(self, faiss_env):
        write_chunks([make_chunk(i, f"t{i}", one_hot(i)) for i in range(3)])
        store = get_store("test")
        q = np.array(one_hot(1), dtype=np.float32)
        result = store.search(q, k=2)
        assert result[0] == "c1"
        assert len(result) == 2

    def test_search_k_exceeds_ntotal(self, faiss_env):
        write_chunks([make_chunk(0, "t0", one_hot(0))])
        store = get_store("test")
        q = np.array(one_hot(0), dtype=np.float32)
        assert store.search(q, k=10) == ["c0"]

    def test_reconstruct_returns_vector(self, faiss_env):
        write_chunks([make_chunk(0, "t0", one_hot(0))])
        store = get_store("test")
        vec = store.reconstruct("c0")
        assert vec is not None
        # COSINE 写入时已归一化，one-hot 归一化后不变
        np.testing.assert_allclose(vec, np.array(one_hot(0), dtype=np.float32))
        assert store.reconstruct("missing") is None

    def test_all_chunks(self, faiss_env):
        write_chunks([make_chunk(i, f"t{i}", one_hot(i)) for i in range(3)])
        store = get_store("test")
        pairs = store.all_chunks()
        assert ("c0", "t0") in pairs
        assert len(pairs) == 3

    def test_is_empty_and_reload_bumps_version(self, faiss_env):
        write_chunks([make_chunk(0, "t0", one_hot(0))])
        store = get_store("test")
        assert not store.is_empty
        v0 = store.version
        # 追加写入后 reload 可见新数据
        write_chunks([make_chunk(1, "t1", one_hot(1))])
        store.reload()
        assert store.version == v0 + 1
        assert store.get_chunk("c1") is not None

    def test_ivf_index_load(self, faiss_env):
        """IVF 分支：nprobe 设置 + direct map 重建可用"""
        settings.faiss.index_type = "IVF_FLAT"
        write_chunks([make_chunk(i, f"t{i}", one_hot(i % 8, scale=1.0 + i))
                      for i in range(12)])
        store = get_store("test")
        q = np.array(one_hot(3), dtype=np.float32)
        assert len(store.search(q, k=2)) == 2
        assert store.reconstruct("c3") is not None

    def test_get_store_caches_instance(self, faiss_env):
        write_chunks([make_chunk(0, "t0", one_hot(0))])
        assert get_store("test") is get_store("test")
        reset_stores()
        # 重置后是新实例
        assert get_store("test") is not None
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/unit/retrieval/test_store.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'retrieval'`

- [ ] **Step 4: 实现**

创建空的 `src/retrieval/__init__.py`。创建 `src/retrieval/store.py`：

```python
"""FAISSStore — FAISS 索引 + docstore 只读访问（每 collection 一个实例）"""
import json
import threading

import faiss
import numpy as np

from logger import logger
from models.chunk import Chunk


class FAISSStore:
    """单 collection 的 FAISS 索引 + docstore 只读封装

    线程安全懒加载；reload() 热重载并递增 version（BM25 缓存失效依据）。
    """

    def __init__(self, collection: str):
        self.collection = collection
        self.version = 0
        self._lock = threading.Lock()
        self._index = None
        self._docstore: dict = {}     # chunk_id -> entry
        self._id_map: dict = {}       # faiss_id -> chunk_id
        self._loaded = False

    # ---- 加载 ----------------------------------------------------------

    def load(self) -> None:
        """幂等加载；collection 目录不存在抛 ValueError"""
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                self._load_unlocked()

    def _load_unlocked(self) -> None:
        from config import settings

        index_dir = settings.faiss.index_dir / self.collection
        if not index_dir.exists():
            raise ValueError(f"Collection '{self.collection}' 不存在: {index_dir}")

        index_path = index_dir / "index.faiss"
        self._index = None
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            if isinstance(index, faiss.IndexIVFFlat):
                index.nprobe = settings.faiss.nprobe
                index.make_direct_map()   # MMR 需 reconstruct 原始向量
            self._index = index

        self._docstore = {}
        docstore_path = index_dir / "docstore.json"
        if docstore_path.exists():
            with open(docstore_path, encoding="utf-8") as f:
                self._docstore = json.load(f)

        self._id_map = {
            entry["faiss_id"]: cid
            for cid, entry in self._docstore.items()
            if "faiss_id" in entry
        }
        self._loaded = True

    def reload(self) -> None:
        """热重载（索引更新后调用）；version 递增使 BM25 缓存失效"""
        with self._lock:
            self._loaded = False
            self._load_unlocked()
            self.version += 1

    # ---- 查询 ----------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """docstore entry → 新 Chunk 实例（每次新建，防调用方污染）"""
        entry = self._docstore.get(chunk_id)
        if entry is None:
            return None
        return Chunk(
            chunk_id=chunk_id,
            doc_id=entry.get("doc_id", ""),
            text=entry.get("text", ""),
            chunk_index=entry.get("chunk_index", 0),
            prev_chunk_id=entry.get("prev_chunk_id"),
            next_chunk_id=entry.get("next_chunk_id"),
            metadata=dict(entry.get("metadata", {})),
        )

    def search(self, vector: np.ndarray, k: int) -> list[str]:
        """FAISS 向量搜索 → 按相关度降序的 chunk_id 列表"""
        if self.is_empty:
            return []
        k = min(k, self._index.ntotal)
        _, ids = self._index.search(
            vector.reshape(1, -1).astype(np.float32), k
        )
        result = []
        for faiss_id in ids[0]:
            if faiss_id < 0:
                continue
            chunk_id = self._id_map.get(int(faiss_id))
            if chunk_id is None:
                logger.warning(
                    "FAISS id %s 在 docstore 中不存在，已跳过", faiss_id
                )
                continue
            result.append(chunk_id)
        return result

    def reconstruct(self, chunk_id: str) -> np.ndarray | None:
        """取 chunk 的原始向量（MMR 多样性计算用）"""
        entry = self._docstore.get(chunk_id)
        if entry is None or "faiss_id" not in entry or self._index is None:
            return None
        try:
            return self._index.reconstruct(int(entry["faiss_id"]))
        except RuntimeError:
            return None

    def all_chunks(self) -> list[tuple[str, str]]:
        """(chunk_id, text) 列表，供 BM25 建索引"""
        return [
            (cid, entry.get("text", ""))
            for cid, entry in self._docstore.items()
        ]


# ---- 模块级 store 缓存 ---------------------------------------------------

_stores: dict[str, FAISSStore] = {}
_stores_lock = threading.Lock()


def get_store(collection: str) -> FAISSStore:
    """获取（并懒加载）collection 对应的 FAISSStore 单例"""
    store = _stores.get(collection)
    if store is None:
        with _stores_lock:
            store = _stores.get(collection)
            if store is None:
                store = FAISSStore(collection)
                _stores[collection] = store
    store.load()
    return store


def reset_stores() -> None:
    """清空 store 缓存（测试用）"""
    with _stores_lock:
        _stores.clear()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/unit/retrieval/test_store.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/ tests/unit/retrieval/
git commit -m "feat(retrieval): add FAISSStore for read-only index and docstore access"
```

---

### Task 3: RRF 融合

**Files:**
- Create: `src/retrieval/fusion.py`
- Test: `tests/unit/retrieval/test_fusion.py`

**Interfaces:**
- Produces: `rrf_fuse(ranked_lists: list[list[str]], rrf_k: int, limit: int) -> list[tuple[str, float]]` — 输入多路召回的有序 chunk_id 列表（位置即 rank，从 1 计），输出按 RRF 分数降序去重的 `(chunk_id, rrf_score)`，截断至 limit

- [ ] **Step 1: 写失败测试**

`tests/unit/retrieval/test_fusion.py`：

```python
"""RRF 融合测试"""
import pytest

from retrieval.fusion import rrf_fuse


class TestRRFFuse:
    def test_score_math(self):
        # a 在两路均排第 1：score = 2/(60+1)；b 两路第 2：2/(60+2)；c 单路第 3：1/(60+3)
        result = rrf_fuse([["a", "b", "c"], ["a", "b"]], rrf_k=60, limit=10)
        assert [cid for cid, _ in result] == ["a", "b", "c"]
        scores = dict(result)
        assert scores["a"] == pytest.approx(2 / 61)
        assert scores["b"] == pytest.approx(2 / 62)
        assert scores["c"] == pytest.approx(1 / 63)

    def test_dedup_across_lists(self):
        result = rrf_fuse([["a"], ["a"], ["a"]], rrf_k=60, limit=10)
        assert len(result) == 1

    def test_limit_truncates(self):
        result = rrf_fuse([["a", "b", "c", "d"]], rrf_k=60, limit=2)
        assert len(result) == 2
        assert result[0][0] == "a"

    def test_empty_input(self):
        assert rrf_fuse([], rrf_k=60, limit=5) == []
        assert rrf_fuse([[], []], rrf_k=60, limit=5) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/retrieval/test_fusion.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现**

`src/retrieval/fusion.py`：

```python
"""RRF 融合去重 — score = Σ 1/(rrf_k + rank)"""


def rrf_fuse(
    ranked_lists: list[list[str]], rrf_k: int, limit: int
) -> list[tuple[str, float]]:
    """多路排名列表 → 按 RRF 分数降序去重合并，截断至 limit

    位置即 rank（从 1 计）。融合层按"多路排名列表"设计，
    后续新增召回路（如摘要索引）直接追加列表即可。
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:limit]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/retrieval/test_fusion.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/fusion.py tests/unit/retrieval/test_fusion.py
git commit -m "feat(retrieval): add RRF fusion with dedup and truncation"
```

---

### Task 4: RetrievalEvaluator — Self-RAG 自评

**Files:**
- Create: `src/retrieval/evaluator.py`
- Test: `tests/unit/retrieval/test_evaluator.py`

**Interfaces:**
- Consumes: `models.enums.RetrievalEval`、`models.chunk.Chunk`（读 `rerank_score`）
- Produces: `evaluate(reranked: list[Chunk]) -> RetrievalEval`

- [ ] **Step 1: 写失败测试**

`tests/unit/retrieval/test_evaluator.py`：

```python
"""Self-RAG 自评测试（默认阈值 sufficient=0.5 / need_more=0.3）"""
from models.chunk import Chunk
from models.enums import RetrievalEval
from retrieval.evaluator import evaluate


def _chunks(scores: list[float]) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", doc_id="d", text="t", chunk_index=i,
              rerank_score=s)
        for i, s in enumerate(scores)
    ]


class TestEvaluate:
    def test_sufficient(self):
        assert evaluate(_chunks([0.8, 0.6])) == RetrievalEval.SUFFICIENT

    def test_sufficient_boundary(self):
        assert evaluate(_chunks([0.5])) == RetrievalEval.SUFFICIENT

    def test_need_more(self):
        assert evaluate(_chunks([0.4, 0.4])) == RetrievalEval.NEED_MORE

    def test_need_more_boundary(self):
        assert evaluate(_chunks([0.3])) == RetrievalEval.NEED_MORE

    def test_insufficient(self):
        assert evaluate(_chunks([0.1, 0.2])) == RetrievalEval.INSUFFICIENT

    def test_empty_is_insufficient(self):
        assert evaluate([]) == RetrievalEval.INSUFFICIENT
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/retrieval/test_evaluator.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现**

`src/retrieval/evaluator.py`：

```python
"""RetrievalEvaluator — Self-RAG 自评：top_k 平均 rerank_score 对照阈值"""
from models.chunk import Chunk
from models.enums import RetrievalEval


def evaluate(reranked: list[Chunk]) -> RetrievalEval:
    """avg(rerank_score) >= 0.5 → SUFFICIENT；>= 0.3 → NEED_MORE；否则 INSUFFICIENT"""
    if not reranked:
        return RetrievalEval.INSUFFICIENT

    from config import settings

    cfg = settings.retrieval
    avg = sum(c.rerank_score for c in reranked) / len(reranked)
    if avg >= cfg.relevance_threshold_sufficient:
        return RetrievalEval.SUFFICIENT
    if avg >= cfg.relevance_threshold_need_more:
        return RetrievalEval.NEED_MORE
    return RetrievalEval.INSUFFICIENT
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/retrieval/test_evaluator.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/evaluator.py tests/unit/retrieval/test_evaluator.py
git commit -m "feat(retrieval): add Self-RAG retrieval evaluator"
```

---

### Task 5: BM25Retriever — 稀疏召回

**Files:**
- Create: `src/retrieval/bm25_retriever.py`
- Test: `tests/unit/retrieval/test_bm25_retriever.py`

**Interfaces:**
- Consumes: store 对象需提供 `all_chunks() -> list[tuple[str, str]]` 和 `version: int`（Task 2 的 `FAISSStore` 满足；测试用 FakeStore）
- Produces: `class BM25Retriever`: `__init__(store)`（构建时全量分词建索引，记录 `self.version = store.version`）、`retrieve(query: str, k: int) -> list[str]`（按 BM25 分数降序 chunk_id，score <= 0 过滤）

- [ ] **Step 1: 写失败测试**

`tests/unit/retrieval/test_bm25_retriever.py`：

```python
"""BM25Retriever 测试（jieba 分词 + rank_bm25 内存索引）"""
from retrieval.bm25_retriever import BM25Retriever


class FakeStore:
    version = 0

    def __init__(self, pairs):
        self._pairs = pairs

    def all_chunks(self):
        return self._pairs


CORPUS = [
    ("c0", "申请年假需要提前三天提交审批"),
    ("c1", "薪资明细可在人事系统查询"),
    ("c2", "差旅报销需提供发票原件"),
]


class TestBM25Retriever:
    def test_chinese_term_match(self):
        r = BM25Retriever(FakeStore(CORPUS))
        result = r.retrieve("年假 审批", k=3)
        assert result[0] == "c0"

    def test_no_match_returns_empty(self):
        r = BM25Retriever(FakeStore(CORPUS))
        assert r.retrieve("量子计算", k=3) == []

    def test_k_truncates(self):
        r = BM25Retriever(FakeStore(CORPUS))
        result = r.retrieve("申请 查询 报销", k=1)
        assert len(result) == 1

    def test_empty_corpus(self):
        r = BM25Retriever(FakeStore([]))
        assert r.retrieve("年假", k=3) == []

    def test_records_store_version(self):
        store = FakeStore(CORPUS)
        store.version = 7
        assert BM25Retriever(store).version == 7
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/retrieval/test_bm25_retriever.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现**

`src/retrieval/bm25_retriever.py`：

```python
"""BM25Retriever — jieba 分词 + rank_bm25 内存稀疏索引

1K~10K 文档规模下启动时从 docstore 全量构建（秒级），不持久化。
"""
import jieba
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class BM25Retriever:
    """构建时记录 store.version，供上层判断索引热重载后是否需要重建"""

    def __init__(self, store):
        self.version = store.version
        pairs = store.all_chunks()
        self._chunk_ids = [cid for cid, _ in pairs]
        corpus = [_tokenize(text) for _, text in pairs]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(self, query: str, k: int) -> list[str]:
        """按 BM25 分数降序返回 chunk_id；score <= 0 的不返回"""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return [self._chunk_ids[i] for i in ranked[:k] if scores[i] > 0]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/retrieval/test_bm25_retriever.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/bm25_retriever.py tests/unit/retrieval/test_bm25_retriever.py
git commit -m "feat(retrieval): add BM25 retriever with in-memory jieba index"
```

---

### Task 6: VectorRetriever — 向量召回

**Files:**
- Create: `src/retrieval/vector_retriever.py`
- Test: `tests/unit/retrieval/test_vector_retriever.py`

**Interfaces:**
- Consumes: store 对象需提供 `search(vector, k) -> list[str]`；encoder 对象需提供 `encode(texts: list[str]) -> ndarray`（SentenceTransformer 兼容）
- Produces:
  - `class VectorRetriever`: `__init__(store, encoder)`、`retrieve(query: str, k: int) -> list[str]`
  - `load_embedding_model()` — 从 `models.get_path("embedding")` 加载 SentenceTransformer，进程内缓存；未下载抛 `RuntimeError`（Task 9 懒加载用）

- [ ] **Step 1: 写失败测试**

`tests/unit/retrieval/test_vector_retriever.py`：

```python
"""VectorRetriever 测试"""
import numpy as np

from retrieval.vector_retriever import VectorRetriever


class FakeStore:
    def __init__(self, result):
        self.result = result
        self.last_vector = None
        self.last_k = None

    def search(self, vector, k):
        self.last_vector = vector
        self.last_k = k
        return self.result[:k]


class FakeEncoder:
    def encode(self, texts):
        # 固定返回 norm=2 的向量，验证 COSINE 归一化
        return np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)


class TestVectorRetriever:
    def test_retrieve_returns_store_result(self):
        store = FakeStore(["c1", "c0"])
        r = VectorRetriever(store, FakeEncoder())
        assert r.retrieve("查询", k=2) == ["c1", "c0"]
        assert store.last_k == 2

    def test_cosine_normalizes_query_vector(self):
        store = FakeStore(["c0"])
        r = VectorRetriever(store, FakeEncoder())
        r.retrieve("查询", k=1)
        # metric_type 默认 COSINE：norm=2 的向量应被归一化为单位向量
        np.testing.assert_allclose(
            store.last_vector, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/retrieval/test_vector_retriever.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现**

`src/retrieval/vector_retriever.py`：

```python
"""VectorRetriever — 查询编码 + FAISS 向量召回"""
import threading

import faiss
import numpy as np

_embedding_model = None
_model_lock = threading.Lock()


def load_embedding_model():
    """从本地路径加载 SentenceTransformer（进程内缓存）

    未下载时抛 RuntimeError 并提示下载命令，不自动触发下载。
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _model_lock:
        if _embedding_model is None:
            from config import settings
            from model import models

            path = models.get_path("embedding")
            if path is None:
                raise RuntimeError(
                    "Embedding 模型未下载，请先执行 "
                    "`from model import models; models.download('embedding')`"
                )
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer(
                str(path), device=settings.embedding.device
            )
    return _embedding_model


class VectorRetriever:
    """查询 → encoder 编码（COSINE 时归一化，与写入侧一致）→ FAISS 搜索"""

    def __init__(self, store, encoder):
        self._store = store
        self._encoder = encoder

    def retrieve(self, query: str, k: int) -> list[str]:
        from config import settings

        vec = np.asarray(self._encoder.encode([query]), dtype=np.float32)
        if settings.faiss.metric_type == "COSINE":
            faiss.normalize_L2(vec)
        return self._store.search(vec[0], k)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/retrieval/test_vector_retriever.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/vector_retriever.py tests/unit/retrieval/test_vector_retriever.py
git commit -m "feat(retrieval): add vector retriever with lazy embedding model loading"
```

---

### Task 7: ContextExpander — 上下文扩展

**Files:**
- Create: `src/retrieval/expander.py`
- Test: `tests/unit/retrieval/test_expander.py`

**Interfaces:**
- Consumes: store 对象需提供 `get_chunk(chunk_id) -> Chunk | None`
- Produces: `class ContextExpander`: `__init__(store)`、`expand(chunk: Chunk, window: int) -> Chunk`（原地改写 `chunk.text` 为窗口拼接文本，`chunk.metadata["window_chunk_ids"]` 记录窗口内 chunk_id，返回同一实例）

- [ ] **Step 1: 写失败测试**

`tests/unit/retrieval/test_expander.py`：

```python
"""ContextExpander 测试"""
from dataclasses import replace

from models.chunk import Chunk
from retrieval.expander import ContextExpander


def _chunk(i: int, prev_id=None, next_id=None) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}", doc_id="d", text=f"t{i}", chunk_index=i,
        prev_chunk_id=prev_id, next_chunk_id=next_id,
    )


class FakeStore:
    def __init__(self, chunks):
        self._m = {c.chunk_id: c for c in chunks}

    def get_chunk(self, cid):
        c = self._m.get(cid)
        return replace(c, metadata=dict(c.metadata)) if c else None


# 链：c0 <-> c1 <-> c2 <-> c3
CHAIN = [
    _chunk(0, next_id="c1"),
    _chunk(1, prev_id="c0", next_id="c2"),
    _chunk(2, prev_id="c1", next_id="c3"),
    _chunk(3, prev_id="c2"),
]


class TestContextExpander:
    def test_window_1_both_sides(self):
        store = FakeStore(CHAIN)
        c = store.get_chunk("c1")
        result = ContextExpander(store).expand(c, window=1)
        assert result is c
        assert c.text == "t0\nt1\nt2"
        assert c.metadata["window_chunk_ids"] == ["c0", "c1", "c2"]

    def test_window_2(self):
        store = FakeStore(CHAIN)
        c = store.get_chunk("c2")
        ContextExpander(store).expand(c, window=2)
        assert c.text == "t0\nt1\nt2\nt3"

    def test_doc_boundary_head(self):
        store = FakeStore(CHAIN)
        c = store.get_chunk("c0")
        ContextExpander(store).expand(c, window=1)
        assert c.text == "t0\nt1"
        assert c.metadata["window_chunk_ids"] == ["c0", "c1"]

    def test_missing_neighbor_skipped(self):
        broken = _chunk(9, next_id="ghost")
        store = FakeStore([broken])
        c = store.get_chunk("c9")
        ContextExpander(store).expand(c, window=1)
        assert c.text == "t9"
        assert c.metadata["window_chunk_ids"] == ["c9"]

    def test_window_0_noop_text(self):
        store = FakeStore(CHAIN)
        c = store.get_chunk("c1")
        ContextExpander(store).expand(c, window=0)
        assert c.text == "t1"
        assert c.metadata["window_chunk_ids"] == ["c1"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/retrieval/test_expander.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现**

`src/retrieval/expander.py`：

```python
"""ContextExpander — 沿 prev/next chunk_id 拼接窗口文本

相邻命中窗口重叠导致的重复文本由 generation 组装层去重，本模块不处理。
"""
from models.chunk import Chunk


class ContextExpander:
    def __init__(self, store):
        self._store = store

    def expand(self, chunk: Chunk, window: int) -> Chunk:
        """向左右各拉 window 个邻居，按 chunk_index 顺序拼接写回 chunk.text"""
        before: list[Chunk] = []
        cur = chunk
        for _ in range(window):
            if not cur.prev_chunk_id:
                break
            prev = self._store.get_chunk(cur.prev_chunk_id)
            if prev is None:
                break
            before.insert(0, prev)
            cur = prev

        after: list[Chunk] = []
        cur = chunk
        for _ in range(window):
            if not cur.next_chunk_id:
                break
            nxt = self._store.get_chunk(cur.next_chunk_id)
            if nxt is None:
                break
            after.append(nxt)
            cur = nxt

        window_chunks = before + [chunk] + after
        chunk.metadata["window_chunk_ids"] = [c.chunk_id for c in window_chunks]
        chunk.text = "\n".join(c.text for c in window_chunks)
        return chunk
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/retrieval/test_expander.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/expander.py tests/unit/retrieval/test_expander.py
git commit -m "feat(retrieval): add context expander via prev/next chunk window"
```

---

### Task 8: Reranker — CrossEncoder 精排 + MMR

**Files:**
- Create: `src/retrieval/reranker.py`
- Test: `tests/unit/retrieval/test_reranker.py`

**Interfaces:**
- Consumes: cross_encoder 对象需提供 `predict(pairs: list[tuple[str, str]]) -> ndarray`（sentence-transformers `CrossEncoder` 兼容；BGE reranker num_labels=1 时 predict 默认带 sigmoid，输出 0~1）
- Produces:
  - `class Reranker`: `__init__(cross_encoder)`、`rerank(query: str, chunks: list[Chunk]) -> list[Chunk]`（写 `c.rerank_score`，按分数降序返回）
  - `mmr_select(chunks: list[Chunk], vectors: dict[str, np.ndarray | None], top_k: int, mmr_lambda: float) -> list[Chunk]`（纯函数：相关性 = rerank_score，多样性 = 向量余弦；缺失向量按相似度 0 处理）
  - `load_cross_encoder()` — 从 `models.get_path("rerank")` 加载 CrossEncoder，进程内缓存；未下载抛 `RuntimeError`

- [ ] **Step 1: 写失败测试**

`tests/unit/retrieval/test_reranker.py`：

```python
"""Reranker（CrossEncoder 精排 + MMR）测试"""
import numpy as np

from models.chunk import Chunk
from retrieval.reranker import Reranker, mmr_select


def _chunk(cid: str, text: str = "", score: float = 0.0) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", text=text, chunk_index=0,
                 rerank_score=score)


class MockCrossEncoder:
    """含"年假"的文本高分，其余低分"""

    def predict(self, pairs):
        return np.array(
            [0.9 if "年假" in text else 0.2 for _, text in pairs]
        )


class TestReranker:
    def test_scores_written_and_sorted(self):
        chunks = [_chunk("c0", "报销流程"), _chunk("c1", "年假申请")]
        result = Reranker(MockCrossEncoder()).rerank("年假", chunks)
        assert [c.chunk_id for c in result] == ["c1", "c0"]
        assert result[0].rerank_score == 0.9
        assert result[1].rerank_score == 0.2

    def test_empty_chunks(self):
        assert Reranker(MockCrossEncoder()).rerank("q", []) == []


class TestMMRSelect:
    # a/b 向量相同（冗余），c 正交（多样）
    VECTORS = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([1.0, 0.0], dtype=np.float32),
        "c": np.array([0.0, 1.0], dtype=np.float32),
    }

    def _chunks(self):
        return [
            _chunk("a", score=0.9),
            _chunk("b", score=0.8),
            _chunk("c", score=0.7),
        ]

    def test_lambda_1_pure_relevance(self):
        result = mmr_select(self._chunks(), self.VECTORS, top_k=2, mmr_lambda=1.0)
        assert [c.chunk_id for c in result] == ["a", "b"]

    def test_lambda_0_pure_diversity(self):
        # 首选最高分 a；之后 b 与 a 相似度 1、c 与 a 相似度 0 → 选 c
        result = mmr_select(self._chunks(), self.VECTORS, top_k=2, mmr_lambda=0.0)
        assert [c.chunk_id for c in result] == ["a", "c"]

    def test_pool_smaller_than_top_k(self):
        chunks = self._chunks()
        result = mmr_select(chunks, self.VECTORS, top_k=10, mmr_lambda=0.7)
        assert len(result) == 3
        assert result[0].chunk_id == "a"

    def test_missing_vector_treated_as_diverse(self):
        vectors = {"a": np.array([1.0, 0.0], dtype=np.float32),
                   "b": None, "c": None}
        result = mmr_select(self._chunks(), vectors, top_k=2, mmr_lambda=0.0)
        # b 向量缺失 → 相似度按 0 处理，仍按 MMR 得分参与竞争（0 - 0 > 0 - sim(c)? 均为 0，取先者 b）
        assert result[0].chunk_id == "a"
        assert len(result) == 2

    def test_empty(self):
        assert mmr_select([], {}, top_k=5, mmr_lambda=0.7) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/retrieval/test_reranker.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现**

`src/retrieval/reranker.py`：

```python
"""Reranker — CrossEncoder 精排 + MMR 多样性选择

BGE reranker num_labels=1，sentence-transformers CrossEncoder.predict
默认经 Sigmoid 激活输出 0~1，与 relevance_threshold_* 阈值同量纲。
MMR 多样性用原始 chunk 向量（FAISS reconstruct）计算余弦——窗口扩展文本
无现成 embedding，原始向量是足够好的近似（设计文档 5.6 已评审确认）。
"""
import threading

import numpy as np

from models.chunk import Chunk

_cross_encoder = None
_ce_lock = threading.Lock()


def load_cross_encoder():
    """从本地路径加载 CrossEncoder（进程内缓存）；未下载抛 RuntimeError"""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    with _ce_lock:
        if _cross_encoder is None:
            from config import settings
            from model import models

            path = models.get_path("rerank")
            if path is None:
                raise RuntimeError(
                    "Rerank 模型未下载，请先执行 "
                    "`from model import models; models.download('rerank')`"
                )
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder(
                str(path), device=settings.embedding.device
            )
    return _cross_encoder


class Reranker:
    def __init__(self, cross_encoder):
        self._ce = cross_encoder

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """(query, chunk.text) 逐对打分写入 rerank_score，按分数降序返回"""
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._ce.predict(pairs)
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mmr_select(
    chunks: list[Chunk],
    vectors: dict[str, np.ndarray | None],
    top_k: int,
    mmr_lambda: float,
) -> list[Chunk]:
    """MMR 贪心：score = λ·rerank_score - (1-λ)·max_sim(已选)"""
    if not chunks:
        return []
    pool = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    if len(pool) <= top_k:
        return pool

    selected = [pool.pop(0)]
    while pool and len(selected) < top_k:
        best_idx, best_score = 0, float("-inf")
        for i, c in enumerate(pool):
            max_sim = max(
                _cosine(vectors.get(c.chunk_id), vectors.get(s.chunk_id))
                for s in selected
            )
            score = mmr_lambda * c.rerank_score - (1 - mmr_lambda) * max_sim
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(pool.pop(best_idx))
    return selected
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/retrieval/test_reranker.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/reranker.py tests/unit/retrieval/test_reranker.py
git commit -m "feat(retrieval): add cross-encoder reranker with MMR selection"
```

---

### Task 9: RetrievalLayer 主编排器 + 单例工厂

**Files:**
- Create: `src/retrieval/layer.py`
- Modify: `src/retrieval/__init__.py`（Task 2 建的空文件 → 填充单例工厂）
- Test: `tests/unit/retrieval/test_layer.py`

**Interfaces:**
- Consumes: Task 2-8 全部产出（`get_store` / `rrf_fuse` / `evaluate` / `BM25Retriever` / `VectorRetriever` + `load_embedding_model` / `ContextExpander` / `Reranker` + `mmr_select` + `load_cross_encoder`）；`models.context.PipelineContext`（读 `query` / `rewritten_queries` / `collection`，写 `candidates` / `reranked` / `retrieval_eval` / `metadata`）
- Produces:
  - `class RetrievalLayer`: `__init__(encoder=None, cross_encoder=None)`（None 时首次 retrieve 懒加载真实模型；测试注入 mock）、`async retrieve(ctx: PipelineContext) -> PipelineContext`
  - `get_retrieval_layer() -> RetrievalLayer`、`reset_retrieval_layer() -> None`（同时清空 store 缓存）

- [ ] **Step 1: 写失败测试**

`tests/unit/retrieval/test_layer.py`：

```python
"""RetrievalLayer 编排 + 单例测试（小索引端到端，mock 模型）"""
import numpy as np
import pytest

from models.context import PipelineContext
from models.enums import RetrievalEval
from retrieval import RetrievalLayer, get_retrieval_layer, reset_retrieval_layer
from tests.unit.retrieval.conftest import DIM, make_chunk, one_hot, write_chunks


class MockEncoder:
    """按关键词映射到 one-hot 向量"""

    TOPICS = {"年假": 0, "薪资": 1, "报销": 2}

    def encode(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(DIM, dtype=np.float32)
            for kw, i in self.TOPICS.items():
                if kw in t:
                    v[i] = 1.0
            vecs.append(v)
        return np.array(vecs)


class MockCrossEncoder:
    def predict(self, pairs):
        return np.array(
            [0.9 if "年假" in text else 0.1 for _, text in pairs]
        )


def _write_corpus():
    # d1: c0 <-> c1（年假文档两段，验证扩展）；d2/d3 干扰项
    write_chunks([
        make_chunk(0, "申请年假需提前三天提交审批", one_hot(0), next_id="c1"),
        make_chunk(1, "年假审批需附上假条材料", one_hot(0), prev_id="c0"),
        make_chunk(2, "薪资明细可在人事系统查询", one_hot(1), doc_id="d2"),
        make_chunk(3, "差旅报销需提供发票原件", one_hot(2), doc_id="d3"),
    ])


def _layer() -> RetrievalLayer:
    return RetrievalLayer(encoder=MockEncoder(), cross_encoder=MockCrossEncoder())


class TestRetrievalLayer:
    async def test_end_to_end(self, faiss_env):
        _write_corpus()
        ctx = PipelineContext(query="申请年假需要什么材料？", collection="test")
        ctx.rewritten_queries = ["申请年假需要什么材料？", "年假 材料 审批"]
        ctx = await _layer().retrieve(ctx)

        assert ctx.candidates, "RRF 融合后应有候选"
        assert ctx.reranked
        # 年假相关 chunk 精排最高
        assert "年假" in ctx.reranked[0].text
        assert ctx.reranked[0].rerank_score == pytest.approx(0.9)
        # 上下文扩展：c0 的窗口应包含 c1 文本
        top = ctx.reranked[0]
        assert len(top.metadata["window_chunk_ids"]) >= 2
        # Self-RAG 自评已写入
        assert ctx.retrieval_eval is not None
        # 耗时埋点
        assert "retrieval_recall_ms" in ctx.metadata
        assert "retrieval_rerank_ms" in ctx.metadata

    async def test_no_rewritten_queries_falls_back_to_query(self, faiss_env):
        _write_corpus()
        ctx = PipelineContext(query="年假审批", collection="test")
        ctx = await _layer().retrieve(ctx)
        assert ctx.reranked

    async def test_missing_collection_raises(self, faiss_env):
        ctx = PipelineContext(query="年假", collection="ghost")
        with pytest.raises(ValueError, match="ghost"):
            await _layer().retrieve(ctx)

    async def test_reranked_truncated_to_top_k(self, faiss_env):
        from config import settings

        _write_corpus()
        saved = settings.retrieval.top_k
        settings.retrieval.top_k = 2
        try:
            ctx = PipelineContext(query="年假 薪资 报销", collection="test")
            ctx = await _layer().retrieve(ctx)
            assert len(ctx.reranked) <= 2
        finally:
            settings.retrieval.top_k = saved

    async def test_empty_collection_returns_insufficient(self, faiss_env):
        """目录存在但无索引文件 → 空结果 + INSUFFICIENT，不报错"""
        (faiss_env / "empty").mkdir()
        ctx = PipelineContext(query="年假", collection="empty")
        ctx = await _layer().retrieve(ctx)
        assert ctx.candidates == []
        assert ctx.reranked == []
        assert ctx.retrieval_eval == RetrievalEval.INSUFFICIENT

    async def test_single_path_failure_degrades(self, faiss_env, monkeypatch):
        """BM25 路异常时向量路仍可用（单路降级）"""
        _write_corpus()
        from retrieval.bm25_retriever import BM25Retriever

        def _boom(self, query, k):
            raise RuntimeError("boom")

        monkeypatch.setattr(BM25Retriever, "retrieve", _boom)
        ctx = PipelineContext(query="年假审批", collection="test")
        ctx = await _layer().retrieve(ctx)
        assert ctx.reranked, "向量单路仍应产出结果"


class TestSingleton:
    def test_get_returns_same_instance(self):
        reset_retrieval_layer()
        try:
            assert get_retrieval_layer() is get_retrieval_layer()
        finally:
            reset_retrieval_layer()

    def test_reset_creates_new_instance(self):
        reset_retrieval_layer()
        a = get_retrieval_layer()
        reset_retrieval_layer()
        assert get_retrieval_layer() is not a
        reset_retrieval_layer()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/retrieval/test_layer.py -v`
Expected: FAIL，`ImportError: cannot import name 'RetrievalLayer'`

- [ ] **Step 3: 实现 layer.py**

`src/retrieval/layer.py`：

```python
"""RetrievalLayer — 检索层主编排器

召回(向量+BM25 并行) → RRF 融合去重 → 上下文扩展 → 精排+MMR → Self-RAG 自评
"""
import asyncio
import time

from logger import logger
from models.context import PipelineContext
from models.enums import RetrievalEval
from retrieval.bm25_retriever import BM25Retriever
from retrieval.evaluator import evaluate
from retrieval.expander import ContextExpander
from retrieval.fusion import rrf_fuse
from retrieval.reranker import Reranker, load_cross_encoder, mmr_select
from retrieval.store import FAISSStore, get_store
from retrieval.vector_retriever import VectorRetriever, load_embedding_model


class RetrievalLayer:
    """encoder/cross_encoder 为 None 时首次 retrieve 懒加载真实模型（测试注入 mock）"""

    def __init__(self, encoder=None, cross_encoder=None):
        self._encoder = encoder
        self._cross_encoder = cross_encoder
        self._bm25_cache: dict[str, BM25Retriever] = {}

    # ---- 懒加载 --------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = load_embedding_model()
        return self._encoder

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = load_cross_encoder()
        return self._cross_encoder

    def _get_bm25(self, store: FAISSStore) -> BM25Retriever:
        """按 collection 缓存；store 热重载（version 变化）后重建"""
        cached = self._bm25_cache.get(store.collection)
        if cached is None or cached.version != store.version:
            cached = BM25Retriever(store)
            self._bm25_cache[store.collection] = cached
        return cached

    # ---- 主流程 --------------------------------------------------------

    @staticmethod
    def _safe_retrieve(retriever, query: str, k: int, path: str) -> list[str]:
        """单路召回失败降级为空结果，另一路继续"""
        try:
            return retriever.retrieve(query, k)
        except Exception as e:
            logger.error("召回路 [%s] 失败: %s", path, e)
            return []

    async def retrieve(self, ctx: PipelineContext) -> PipelineContext:
        from config import settings

        cfg = settings.retrieval
        loop = asyncio.get_running_loop()

        # store 加载（faiss IO）与 BM25 构建/模型加载均为重活，走线程池
        store = await loop.run_in_executor(None, get_store, ctx.collection)
        if store.is_empty:
            ctx.candidates, ctx.reranked = [], []
            ctx.retrieval_eval = RetrievalEval.INSUFFICIENT
            return ctx

        encoder = await loop.run_in_executor(None, self._get_encoder)
        bm25 = await loop.run_in_executor(None, self._get_bm25, store)
        vector = VectorRetriever(store, encoder)

        # 1. 每条 query 并行两路召回，每路 top_k×2
        queries = ctx.rewritten_queries or [ctx.query]
        recall_k = cfg.top_k * 2
        t0 = time.perf_counter()
        tasks = []
        for q in queries:
            tasks.append(loop.run_in_executor(
                None, self._safe_retrieve, vector, q, recall_k, "vector"))
            tasks.append(loop.run_in_executor(
                None, self._safe_retrieve, bm25, q, recall_k, "bm25"))
        ranked_lists = list(await asyncio.gather(*tasks))
        ctx.metadata["retrieval_recall_ms"] = (time.perf_counter() - t0) * 1000

        # 2. RRF 融合去重 + 截断 → candidates
        fused = rrf_fuse(ranked_lists, cfg.rrf_k, cfg.max_rerank_candidates)
        candidates = []
        for chunk_id, score in fused:
            c = store.get_chunk(chunk_id)
            if c is None:
                logger.warning("chunk %s 在 docstore 中不存在，已跳过", chunk_id)
                continue
            c.metadata["rrf_score"] = score
            candidates.append(c)
        ctx.candidates = candidates

        # 3. 上下文扩展（docstore 内存读，无需线程池）
        t1 = time.perf_counter()
        expander = ContextExpander(store)
        for c in candidates:
            expander.expand(c, cfg.expansion_window)
        ctx.metadata["retrieval_expand_ms"] = (time.perf_counter() - t1) * 1000

        # 4. CrossEncoder 精排（对融合后的标准问法 ctx.query）+ MMR 截断
        t2 = time.perf_counter()
        reranker = Reranker(self._get_cross_encoder())
        reranked = await loop.run_in_executor(
            None, reranker.rerank, ctx.query, candidates
        )
        vectors = {c.chunk_id: store.reconstruct(c.chunk_id) for c in reranked}
        ctx.reranked = mmr_select(reranked, vectors, cfg.top_k, cfg.mmr_lambda)
        ctx.metadata["retrieval_rerank_ms"] = (time.perf_counter() - t2) * 1000

        # 5. Self-RAG 自评
        ctx.retrieval_eval = evaluate(ctx.reranked)
        return ctx
```

- [ ] **Step 4: 实现 __init__.py 单例工厂**

覆写 `src/retrieval/__init__.py`：

```python
"""检索层 — 混合召回 / RRF 融合 / 上下文扩展 / Rerank+MMR / Self-RAG 自评"""
import threading

from retrieval.layer import RetrievalLayer
from retrieval.store import reset_stores

# 全局单例
_retrieval_layer: RetrievalLayer | None = None
_lock = threading.Lock()


def get_retrieval_layer() -> RetrievalLayer:
    """获取检索层全局单例（模型懒加载，首次 retrieve 时初始化）"""
    global _retrieval_layer

    # 快速路径：已初始化，无锁检查
    if _retrieval_layer is not None:
        return _retrieval_layer

    with _lock:
        # 双重检查：可能另一个线程刚完成初始化
        if _retrieval_layer is None:
            _retrieval_layer = RetrievalLayer()
        return _retrieval_layer


def reset_retrieval_layer() -> None:
    """重置全局单例并清空 store 缓存（测试用）"""
    global _retrieval_layer
    with _lock:
        _retrieval_layer = None
        reset_stores()


__all__ = [
    "RetrievalLayer",
    "get_retrieval_layer",
    "reset_retrieval_layer",
]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/unit/retrieval/test_layer.py -v`
Expected: 8 passed

- [ ] **Step 6: 跑全量 retrieval 测试防回归**

Run: `pytest tests/unit/retrieval/ tests/unit/config/ -v`
Expected: 全部 passed

- [ ] **Step 7: Commit**

```bash
git add src/retrieval/ tests/unit/retrieval/test_layer.py
git commit -m "feat(retrieval): add RetrievalLayer orchestrator with singleton factory"
```

---

### Task 10: 文档更新 + 全量验证

**Files:**
- Modify: `CLAUDE.md`（项目结构 retrieval 行 + 当前开发阶段 + 新增"检索模块"用法节）
- Test: 全量测试套件

**Interfaces:**
- Consumes: Task 9 的 `get_retrieval_layer` API

- [ ] **Step 1: 更新 CLAUDE.md 项目结构标记**

`CLAUDE.md` 项目结构中的一行：

```
│   ├── retrieval/             # ⬜ 混合检索 + Rerank + 检索评估
```

改为：

```
│   ├── retrieval/             # ✅ 混合检索（向量+BM25+RRF）+ Rerank + Self-RAG 自评
```

"当前开发阶段"一节：

```
第1期（基础 + 查询理解）已完成：config / security / logger / model / models / session / query。
第2期（检索 + 生成 + 兜底）和后续阶段待实现。
```

改为：

```
第1期（基础 + 查询理解）已完成：config / security / logger / model / models / session / query。
第2期进行中：retrieval 已完成；生成 + 兜底待实现。
```

- [ ] **Step 2: 在 CLAUDE.md "开发要点"中"会话模块"之前插入用法节**

```markdown
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
```

- [ ] **Step 3: 全量测试 + 代码检查**

Run: `pytest tests/ -x -q`
Expected: 全部 passed（含既有 config/security/logger/model/session/query/ingestion 测试，确认无回归）

Run: `ruff check src/retrieval tests/unit/retrieval tests/unit/config`
Expected: All checks passed!（如报 import 排序等问题按提示 `ruff check --fix` 修复后重跑测试）

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: mark retrieval module complete and add usage guide"
```

---

## 计划外事项（不做）

- 摘要索引召回路 / 离线评测（recall@k、MRR）/ Milvus 迁移 / 意图路由 → 见设计文档 §10
- 真实 BGE 模型端到端验证：实现完成后由用户手动执行（需先 `models.download("embedding")` + `models.download("rerank")`），不进 CI
