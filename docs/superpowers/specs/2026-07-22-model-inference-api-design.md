# Model 模块统一推理接口设计

日期: 2026-07-22 | 状态: approved

## 目标

为 `model` 模块新增统一的本地模型推理接口（encode / rerank / generate），消除各业务模块分散加载模型的冗余代码。

## 架构

新增 `src/model/inference.py`，负责模型实例缓存和推理逻辑：

```
model/
├── __init__.py          # 导出 models（不变）
├── manager.py           # 添加推理方法 + 模型属性，委托给 inference.py
├── downloader.py        # 不变
├── inference.py          ← 新增
└── finetune/            # 不变
```

- **`inference.py`** — 模型加载、进程级双检锁缓存、`encode()` / `rerank()` / `generate()` 核心实现
- **`manager.py`** — 对外暴露 `models.encode()` / `models.rerank()` / `models.generate()` 以及 `models.embedding_model` / `models.cross_encoder` 属性

## API 设计

### 模型实例属性

```python
models.embedding_model   # → SentenceTransformer（首次访问懒加载）
models.cross_encoder     # → CrossEncoder（首次访问懒加载）
```

用于需持有模型引用的调用方（VectorRetriever、Reranker、EmbedderStage、SemanticChunker）。

### 推理方法

```python
# Embedding — 透传 SentenceTransformer.encode()
models.encode(["文本1", "文本2"])              # → np.ndarray  shape=(2, 1024)
models.encode("单条文本")                      # → np.ndarray  shape=(1024,)
models.encode(["文本"], batch_size=32, ...)    # **kwargs 透传底层

# Rerank — 透传 CrossEncoder.rank()
models.rerank("query", ["doc1", "doc2"])       # → [{"corpus_id": 0, "score": 0.92}, ...]
models.rerank("query", docs, top_k=5, ...)     # **kwargs 透传底层

# LLM — 预留接口
models.generate("prompt")  # → NotImplementedError
```

### generate() 错误说明

```
generate() 尚未实现。推荐方案：llama-cpp-python + GGUF 量化模型。
  - CPU 友好，内存占用低（INT4 量化后约 4-8 GB）
  - 安装: pip install llama-cpp-python
  - 使用: from llama_cpp import Llama; llm = Llama(model_path="model.gguf")
  - 项目当前 LLM 生成走云端 DeepSeek API，本地推理作为后续迭代方向。
```

## 迁移计划

### 删除

| 文件 | 内容 |
|------|------|
| `retrieval/vector_retriever.py` | `load_embedding_model()` 函数 + 全局变量 `_embedding_model` / `_model_lock` |
| `retrieval/reranker.py` | `load_cross_encoder()` 函数 + 全局变量 `_cross_encoder` / `_ce_lock` |
| `ingestion/__init__.py` | `_cached_embedding_model` / `_model_load_lock` + 直接 `SentenceTransformer()` 构造 |

### 修改

| 文件 | 旧 | 新 |
|------|-----|-----|
| `retrieval/layer.py` | `from retrieval.vector_retriever import load_embedding_model` | 删除 import |
| `retrieval/layer.py` | `from retrieval.reranker import load_cross_encoder` | 删除 import |
| `retrieval/layer.py` | `self._encoder = load_embedding_model()` | `self._encoder = models.embedding_model` |
| `retrieval/layer.py` | `self._cross_encoder = load_cross_encoder()` | `self._cross_encoder = models.cross_encoder` |
| `ingestion/__init__.py` | `SentenceTransformer(str(model_path))` + 手动缓存 | `models.embedding_model` |

### 不需修改

- `retrieval/vector_retriever.py` 中 `VectorRetriever` 类本身 — 继续接收 `encoder` 参数
- `retrieval/reranker.py` 中 `Reranker` 类本身 — 继续接收 `cross_encoder` 参数
- `ingestion/chunker.py` / `ingestion/embedder.py` — 继续接收 `embedding_model` 参数

## 测试计划

### 新增: `tests/unit/model/test_inference.py`

| 用例 | 覆盖点 |
|------|--------|
| `test_encode_single_text` | 单条文本 → `np.ndarray` 1D |
| `test_encode_multiple_texts` | 多条文本 → `np.ndarray` 2D |
| `test_encode_kwargs_passthrough` | kwarg 正确透传底层 `encode()` |
| `test_rerank_returns_ranked_list` | 返回 `list[dict]` 含 corpus_id + score |
| `test_rerank_kwargs_passthrough` | kwarg 正确透传底层 `rank()` |
| `test_generate_raises_not_implemented` | 抛 NotImplementedError，消息含 `llama-cpp-python` / `GGUF` |
| `test_model_not_downloaded_raises` | 未下载时 encode/rerank 抛 RuntimeError |
| `test_model_instance_cached` | 连续两次访问返回同一实例 |
| `test_concurrent_load_once` | 多线程并发首次访问只加载一次 |

### 修改: `tests/unit/ingestion/test_init.py`

- 不再访问 `ingestion._cached_embedding_model`
- 改为通过 `models.embedding_model` 验证线程安全

### 不需修改

- `tests/unit/retrieval/` — 通过 mock `encoder` / `cross_encoder` 参数注入，不引用 `load_*` 函数
