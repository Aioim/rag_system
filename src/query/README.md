# Query 模块 — 查询理解层

## 模块概述

查询理解层是 RAG 在线 Pipeline 的入口，负责将用户原始 Query 转换为精确的检索输入。

- **意图分类**：concept / procedure / compare / lookup + 清晰度判断
- **多轮融合**：指代消解 + 追问补全（基于会话历史）
- **查询改写**：并行执行（HyDE / 同义变体 / BM25 关键词），合并去重

## 文件结构

```
query/
├── __init__.py            # 导出 + get_query_layer 单例工厂
├── layer.py               # QueryUnderstandingLayer — 主编排器
├── intent_classifier.py   # IntentClassifier — 意图分类 + 清晰度
├── context_fuser.py       # ContextFuser — 多轮指代消解
└── rewriters/
    ├── __init__.py        # QueryRewriter 编排器（并行+合并去重）
    ├── base.py            # BaseRewriter 基类（模板方法）
    ├── hyde.py            # HyDERewriter — 生成假设答案
    ├── synonym.py         # SynonymRewriter — 生成同义变体
    └── keyword_rewriter.py # KeywordRewriter — BM25 关键词提取
```

## 快速开始

```python
from query import get_query_layer, reset_query_layer

# 初始化（LLM + SessionManager 注入）
layer = get_query_layer(llm, session_manager)

# 基础查询
ctx = await layer.process("什么是RAG？")
# → ctx.intent = Intent.CONCEPT
# → ctx.rewritten_queries = ["RAG 定义", "检索增强生成概念", ...]

# 多轮对话（自动指代消解 + 追问补全）
ctx = await layer.process("需要什么材料？", session_id="s1")
# → ctx.query = "申请年假需要什么材料？"

# 特定知识库
ctx = await layer.process("配置手册", collection="tech_docs")

# 模糊问题短路
ctx = await layer.process("帮帮我")
# → ctx.needs_clarification = True
# → ctx.clarification_question = "您想了解哪方面内容？"

reset_query_layer()  # 测试用重置
```

## Pipeline 流程

```
用户 Query → 别名映射 → 意图分类+清晰度 → 多轮融合 → 查询改写(并行)
                                                          ├─ HyDE 假设答案
                                                          ├─ 同义变体
                                                          └─ BM25 关键词
                                              → 合并去重 → ctx.rewritten_queries
```

## 组件温度约定

| 组件 | temperature | 原因 |
|------|-------------|------|
| IntentClassifier | 0 | 意图分类需确定性 |
| ContextFuser | 0 | 指代消解需确定性 |
| KeywordRewriter | 0 | BM25 关键词需幂等 |
| HyDERewriter | 0.3 | 假设答案需受控创意 |
| SynonymRewriter | 0.3 | 同义变体需多样性 |

## 配置

查询理解层通过注入的 LLM 实例工作，无需独立配置段。LLM 温度由 `settings.llm.temperatures` 按意图自动选取。
