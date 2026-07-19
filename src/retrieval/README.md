# Retrieval 模块 — 混合检索 + Rerank

## 模块概述

检索层负责从向量数据库和 BM25 索引中召回相关知识块，并通过精排和多样性控制选出最终 Top-K。

- **多路召回**：向量检索（FAISS）+ BM25 关键词检索（jieba 分词）
- **RRF 融合**：Reciprocal Rank Fusion 去重截断
- **上下文扩展**：prev/next chunk 滑窗补偿
- **精排**：CrossEncoder Rerank + MMR 多样性
- **自评**：Self-RAG 检索质量评估（SUFFICIENT / NEED_MORE / INSUFFICIENT）

## 文件结构

```
retrieval/
├── __init__.py           # 导出 + get_retrieval_layer 单例工厂
├── layer.py              # RetrievalLayer — 主编排器
├── vector_retriever.py   # VectorRetriever — FAISS 向量检索
├── bm25_retriever.py     # BM25Retriever — jieba 分词 BM25 检索
├── fusion.py             # RRF 融合 + 去重截断
├── expander.py           # ContextExpander — prev/next 分块上下文扩展
├── reranker.py           # CrossEncoderReranker — 精排 + MMR 多样性
├── evaluator.py          # SelfRAGEvaluator — 检索质量自评
└── store.py              # FAISSStore + BM25Index — 索引读写
```

## 快速开始

```python
from retrieval import get_retrieval_layer, reset_retrieval_layer

layer = get_retrieval_layer()      # 单例；embedding/rerank 模型首次检索时懒加载

ctx = await layer.retrieve(ctx)    # 输入 query 层产出的 PipelineContext
# → ctx.candidates       粗召回（向量+BM25 → RRF 融合去重截断）
# → ctx.reranked         CrossEncoder 精排 + MMR 后的最终 top_k
# → ctx.retrieval_eval   SUFFICIENT / NEED_MORE / INSUFFICIENT

reset_retrieval_layer()            # 测试用重置（同时清空 store 缓存）
```

## Pipeline 流程

```
ctx.rewritten_queries (每条 rewrite)
  ├── 向量检索 (top_k×2) ──┐
  └── BM25 关键词检索 ─────┤
                            ↓
                     RRF 融合去重（截断至 max_rerank_candidates）
                            ↓
                     prev/next 上下文扩展 (expansion_window)
                            ↓
                     CrossEncoder 精排 + MMR
                            ↓
                     Self-RAG 自评 → ctx.retrieval_eval
```

BM25 索引启动时从 docstore 内存构建（jieba 分词）；索引更新后调用 `store.reload()` 自动触发 BM25 重建。

## 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `retrieval.top_k` | 5 | 最终返回结果数 |
| `retrieval.expansion_window` | 1 | 上下文扩展窗口（左右各 N 块） |
| `retrieval.rrf_k` | 60 | RRF 融合参数 |
| `retrieval.max_rerank_candidates` | 30 | 精排候选上限 |
| `retrieval.mmr_lambda` | 0.7 | MMR 多样性控制 |
| `retrieval.relevance_threshold_sufficient` | 0.5 | 结果充分阈值 |
| `retrieval.relevance_threshold_need_more` | 0.3 | 需要补充检索阈值 |

## 依赖

向量检索需要 FAISS 索引已通过 ingestion 离线 Pipeline 构建。
Embedding 和 Reranker 模型通过 `model` 模块下载管理。
