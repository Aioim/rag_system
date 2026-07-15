# Retrieval 模块（检索层）— 设计文档

> 日期：2026-07-15
> 状态：已评审
> 前置：query 模块（查询理解层）、ingestion 模块（FAISS 索引 + docstore）

## 1. 定位

在线 Pipeline 的检索层：接收查询理解层产出的 `PipelineContext`，执行
**两路召回（向量 + BM25）→ RRF 融合 → 上下文扩展 → Cross-Encoder 精排 + MMR → Self-RAG 自评**，
产出写回 `ctx.candidates` / `ctx.reranked` / `ctx.retrieval_eval`。

**边界**（明确不在本模块内）：

- 5.5 上下文组装（跨来源去重、token 预算、Lost-in-the-Middle 缓解）→ generation 模块
- NEED_MORE 补充检索循环、三级兜底 → core / fallback 编排层
- 摘要索引召回路（依赖 chunk 的 `context_summary`，ingestion 尚未生成/持久化该字段）→ 待 Contextual Retrieval 落地后再加，融合层预留多路扩展点
- 离线检索质量评测（recall@k / MRR）→ 待有标注数据后另行立项

## 2. 模块结构

```
src/retrieval/
├── __init__.py          # 导出 + get_retrieval_layer / reset_retrieval_layer
├── layer.py             # RetrievalLayer — 主编排器
├── store.py             # FAISSStore — FAISS 索引 + docstore 加载与缓存
├── vector_retriever.py  # VectorRetriever — 向量召回
├── bm25_retriever.py    # BM25Retriever — jieba + rank_bm25 内存索引
├── fusion.py            # RRF 融合去重
├── expander.py          # ContextExpander — prev/next chunk 窗口扩展
├── reranker.py          # Reranker — CrossEncoder 精排 + MMR 多样性选择
└── evaluator.py         # RetrievalEvaluator — Self-RAG 自评
```

## 3. 对外接口

```python
from retrieval import get_retrieval_layer, reset_retrieval_layer

layer = get_retrieval_layer()      # 单例，懒加载 embedding/rerank 模型
ctx = await layer.retrieve(ctx)    # 输入查询理解层产出的 PipelineContext
# → ctx.candidates      粗召回结果（RRF 融合去重、截断后）
# → ctx.reranked        精排 + MMR + 截断后的最终 top_k
# → ctx.retrieval_eval  SUFFICIENT / NEED_MORE / INSUFFICIENT
# → ctx.metadata        各阶段耗时（retrieval_recall_ms / rerank_ms 等）
```

- `collection` 从 `ctx.collection` 读取；每个 collection 独立一套 FAISS + BM25 索引，按需懒加载并缓存
- 精排对照的 query 是融合后的 `ctx.query`（单一标准问法）；多路 `rewritten_queries` 只用于召回

## 4. 数据流

```
queries = ctx.rewritten_queries or [ctx.query]     # rewriter 输出已含原始查询
    │
    ├── 每条 query 并行跑两路（asyncio.gather，同步推理丢线程池）
    │   ├── VectorRetriever：BGE 编码 → FAISS 搜索 → top_k×2
    │   └── BM25Retriever：jieba 分词 → rank_bm25 → top_k×2
    │
    ├── RRF 融合去重：score = Σ 1/(rrf_k + rank_i)，按 chunk_id 去重
    │       ↓ 截断至 max_rerank_candidates（默认 30）→ ctx.candidates
    ├── 上下文扩展：经 docstore 拉 prev/next 各 expansion_window 个 chunk，
    │   按 chunk_index 顺序拼接为窗口文本写回 c.text
    ├── CrossEncoder 精排：对 (ctx.query, 窗口文本) 打分，
    │   sigmoid 归一化 0~1 → c.rerank_score
    ├── MMR 多样性选择：mmr_lambda 加权贪心，取最终 top_k → ctx.reranked
    └── Self-RAG 自评：avg(top_k rerank_score) 对照 0.5 / 0.3 阈值
                       → ctx.retrieval_eval
```

## 5. 组件设计

### 5.1 FAISSStore — 索引与 docstore 访问

每 collection 一个实例，线程安全懒加载 + 缓存。

- 加载 `data/faiss_indexes/{collection}/index.faiss` + `docstore.json`，构建
  `chunk_id → entry` 与 `faiss_id → chunk_id` 双向映射
- IVF 索引：设置 `nprobe`（来自 `faiss.nprobe` 配置），`make_direct_map()` 以支持
  `reconstruct()`（MMR 取原始向量，避免重新编码）
- 接口：
  - `get_chunk(chunk_id) -> Chunk | None` — docstore entry 组装为 `models.Chunk`
  - `search(vector, k) -> list[(chunk_id, rank)]` — FAISS 搜索
  - `reconstruct(chunk_id) -> np.ndarray | None` — 取原始向量
  - `all_chunks() -> list[(chunk_id, text)]` — 供 BM25 建索引
  - `reload()` — 索引更新后热重载（同时触发 BM25 重建）
- collection 目录不存在 → `ValueError`（API 层映射 4xx）；索引为空 → 空结果，不报错

### 5.2 VectorRetriever — 向量召回

- SentenceTransformer 从 `models.get_path("embedding")` 加载，模块级缓存
- `encode` 提交线程池；`metric_type=COSINE` 时查询向量 `normalize_L2`
  （与 ingestion 写入侧一致）
- 返回 top_k×2 的 `[(chunk_id, rank)]`

### 5.3 BM25Retriever — 稀疏召回

- Store 加载时用 `jieba.lcut` 对全部 chunk 文本分词，内存构建 `rank_bm25.BM25Okapi`
  （1K~10K 文档规模，构建秒级，不改动 ingestion）
- 查询同样 jieba 分词，返回 top_k×2；score ≤ 0 的不返回

### 5.4 RRF 融合

- 输入：多条 query × 两路的排名列表
- `score(c) = Σ 1/(rrf_k + rank_i)`，按 chunk_id 去重合并、降序排序、
  截断至 `max_rerank_candidates`
- RRF 分数记入 `chunk.metadata["rrf_score"]`；融合层接口按"多路排名列表"设计，
  为后续摘要召回路预留扩展点

### 5.5 ContextExpander — 上下文扩展

- 沿 `prev_chunk_id` / `next_chunk_id` 各拉 `expansion_window` 个邻居，
  按 `chunk_index` 顺序拼接为窗口文本写回 `c.text`
- `metadata["window_chunk_ids"]` 记录窗口内全部 chunk_id
- 相邻命中窗口重叠导致的重复文本由 generation 的组装层去重，本模块不处理
- 邻居 chunk_id 在 docstore 查不到 → 跳过该邻居，保留已有部分

### 5.6 Reranker — CrossEncoder 精排 + MMR

- `sentence_transformers.CrossEncoder` 从 `models.get_path("rerank")` 加载，
  模块级缓存；`predict` 带 sigmoid 激活 → 分数归一 0~1
  （与 `relevance_threshold_*` 阈值同一量纲），批量推理提交线程池
- MMR：相关性用 `rerank_score`，多样性用 FAISS `reconstruct` 的**原始 chunk 向量**
  算余弦（窗口文本无现成 embedding，重新编码代价高，原始向量是足够好的近似——已评审确认）
- `mmr_lambda`（默认 0.7）加权贪心选出最终 top_k

### 5.7 RetrievalEvaluator — Self-RAG 自评

- `avg(top_k rerank_score)` 对照阈值：
  - `>= relevance_threshold_sufficient (0.5)` → `SUFFICIENT`
  - `>= relevance_threshold_need_more (0.3)` → `NEED_MORE`
  - 其余 → `INSUFFICIENT`
- `reranked` 为空 → 直接 `INSUFFICIENT`

### 5.8 RetrievalLayer — 主编排器

- `async def retrieve(ctx: PipelineContext) -> PipelineContext`
- 召回并行度：query × 路 二维展开后 `asyncio.gather`
- 各阶段耗时写入 `ctx.metadata`（命名风格同 ingestion：`{stage}_ms`）

## 6. 全局单例

```python
# src/retrieval/__init__.py
get_retrieval_layer() -> RetrievalLayer    # 双重检查锁单例（同 get_query_layer 模式）
reset_retrieval_layer()                    # 测试用重置
```

## 7. 配置与依赖

**配置**：复用现有 `retrieval.*` 全部键，新增一项（`config/defaults.yaml` +
`src/config/settings.py` 的 `RetrievalConfig`）：

```yaml
retrieval:
  max_rerank_candidates: 30   # RRF 后进入精排的候选上限（CPU CrossEncoder 延迟保护）
```

理由：最坏情况 6 条 query × 2 路 × top_k×2 去重后可达上百候选，CPU 上
Cross-Encoder 会拖到数秒级；30 为 top_k=5 的 6 倍余量。

**依赖**：`pyproject.toml` 的 `[retrieval]` 可选组已有
`sentence-transformers` / `rank-bm25`，补充 `jieba>=0.42`（中文分词，BM25 必需）。

**模型加载**：通过 `models.get_path()` 获取本地路径；未下载时抛带指引的错误
（提示 `models.download(...)`），不自动触发下载。模型实例进程内只加载一次。

## 8. 错误处理

| 场景 | 行为 |
|------|------|
| collection 目录不存在 | 抛 `ValueError`，API 层映射 4xx |
| 索引存在但为空 | 空 candidates + `INSUFFICIENT`，不报错 |
| docstore 与 FAISS 不一致（faiss_id 查不到） | 跳过该条 + warning 日志 |
| 单路召回失败（如 BM25 异常） | 记日志、该路返回空，另一路继续（单路降级） |
| embedding / rerank 模型未下载 | 抛错并提示下载命令 |

## 9. 测试策略

- **单测**（假数据，不加载真实模型）：
  - fusion：RRF 数学正确性 / 去重 / 截断
  - expander：窗口拼接 / 文档边界 chunk / 邻居缺失
  - evaluator：三档阈值边界值
  - bm25：中文分词命中
  - MMR：λ 极值行为（λ=1 纯相关性、λ=0 纯多样性）
- **集成测试**：tmp_path 下用 `FAISSIndexWriter` 写小索引
  （小维度向量 + mock embedding / mock CrossEncoder），端到端跑
  `layer.retrieve()`，验证 `candidates` / `reranked` / `retrieval_eval`
- 真实模型端到端验证在实现完成后手动执行（模型体积大，不进 CI）

## 10. 不在本期范围

- 摘要索引召回路（Contextual Retrieval 前置依赖未落地）
- 离线检索质量评测（recall@k / MRR，需标注集）
- Milvus 迁移（Retriever 接口已隔离，迁移时替换 store + retriever 实现）
- 意图 → collection 路由（一期由调用方显式传 collection）
