# Models 模块 — 共享数据模型层

## 模块概述

Models 模块定义 RAG Pipeline 中所有模块共享的数据结构，无运行时依赖，是所有模块的类型基础。

- **PipelineContext** — 全链路上下文容器，各模块通过它传递数据
- **Chunk / Document / Session** — 核心领域实体
- **API 模型** — Chat / Search 请求响应
- **枚举** — Intent / RetrievalEval / FallbackLevel / DocumentStatus

## 文件结构

```
models/
├── __init__.py      # 统一导出
├── context.py       # PipelineContext — RAG Pipeline 数据容器
├── chunk.py         # Chunk — 文档分块（含 embedding、prev/next 链表）
├── document.py      # Document — 原始文档
├── session.py       # Session + Message — 会话与消息
├── enums.py         # Intent / RetrievalEval / FallbackLevel / DocumentStatus
├── llm.py           # LLMProtocol — LLM 客户端协议
├── api.py           # ChatRequest / ChatResponse / SearchRequest / Source
└── json_utils.py    # 向后兼容重导出 → 已迁移至 src/utils/json_utils.py
```

## 核心数据模型

### PipelineContext

```python
from models.context import PipelineContext

ctx = PipelineContext(query="什么是RAG？")
ctx.intent = Intent.CONCEPT                # 查询层产出
ctx.rewritten_queries = ["RAG 定义", ...]  # 查询层产出
ctx.reranked = [chunk1, chunk2, ...]       # 检索层产出
ctx.answer = "RAG 是..."                    # 生成层产出
ctx.sources = [...]                         # 生成层产出
ctx.confidence = 0.85                       # 综合置信度
```

### Chunk

```python
from models.chunk import Chunk

chunk = Chunk(
    chunk_id="c1",
    doc_id="d1",
    content="文档分块文本内容",
    embedding=[0.1, 0.2, ...],
    prev_chunk_id=None,      # 链表：上一块
    next_chunk_id="c2",       # 链表：下一块
    metadata={"page": 1},
)
```

### Session & Message

```python
from models.session import Session, Message

session = Session(
    session_id="abc-123",
    messages=[Message(role="user", content="..."), ...],
    summary="对话摘要",
)
```

## 枚举

| 枚举 | 值 | 说明 |
|------|-----|------|
| `Intent` | CONCEPT / PROCEDURE / COMPARE / LOOKUP | 查询意图 |
| `RetrievalEval` | SUFFICIENT / NEED_MORE / INSUFFICIENT | 检索质量评估 |
| `FallbackLevel` | NONE / PARTIAL / WEB_SEARCH / NO_ANSWER | 兜底级别 |
| `DocumentStatus` | PENDING / DONE / FAILED | 文档处理状态 |

## LLMProtocol

```python
from typing import Protocol
from models.llm import LLMProtocol

# LLM 客户端需满足此协议
class MyLLM:
    async def generate(self, prompt: str, **kwargs) -> str: ...
    async def ainvoke(self, messages: list, **kwargs) -> str: ...
```
