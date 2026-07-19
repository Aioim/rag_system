# Core 模块 — RAG Pipeline 编排

## 模块概述

Core 模块是 RAG 在线 Pipeline 的顶层编排器，串联查询理解 → 检索 → 兜底 → 生成 → 会话记录全链路。

- **全链路编排**：统一调度 query / retrieval / fallback / generation 各层
- **独立降级**：每层异常时记录日志并继续，不中断 Pipeline
- **会话记录**：自动记录用户 Query 和系统 Answer 到 SQLite

## 文件结构

```
core/
├── __init__.py      # 导出 + get_rag_pipeline 单例工厂
├── pipeline.py      # RAGPipeline — 全链路主编排器
└── fallback.py      # FallbackHandler — 薄包装层，委托至 src/fallback/
```

## 快速开始

```python
from core import get_rag_pipeline, reset_rag_pipeline

# 初始化（LLM + SessionManager 注入）
pipeline = get_rag_pipeline(llm, session_manager)

# 执行完整 RAG 问答
ctx = await pipeline.run("什么是RAG？")
# → ctx.answer         LLM 生成的回答（含事实核查标注）
# → ctx.sources        引用来源列表
# → ctx.confidence     置信度
# → ctx.is_fallback    是否触发兜底
# → ctx.fallback_level 兜底级别

# 多轮对话
ctx = await pipeline.run("需要什么材料？", session_id="s1")

# 指定知识库
ctx = await pipeline.run("配置手册", collection="tech_docs")

reset_rag_pipeline()  # 测试用重置
```

## 编排流程

```
用户 Query
  │
  ├── 1. QueryUnderstandingLayer.process()
  │     ├── 别名映射
  │     ├── 意图分类 + 清晰度
  │     ├── 多轮上下文融合
  │     └── 查询改写（并行）
  │
  ├── 2. RetrievalLayer.retrieve()
  │     ├── 向量 + BM25 多路召回（每条改写 query）
  │     ├── RRF 融合 → prev/next 扩展 → Rerank+MMR
  │     └── Self-RAG 自评
  │
  ├── 3. FallbackHandler.handle()（条件触发）
  │     ├── NEED_MORE → 补充检索
  │     └── INSUFFICIENT → 联网搜索 / 诚实告知
  │
  ├── 4. GenerationLayer.generate()
  │     └── Prompt 组装 → 路由 → 生成 → 事实核查 → 引用
  │
  └── 5. session_manager.add_message()（记录问答）
```

## 异常处理

每层包裹 try/except，异常时：
- 记录 ERROR 日志
- 标记 `ctx.errors` 
- 让下游继续处理（能返回什么返回什么）
