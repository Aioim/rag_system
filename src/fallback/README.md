# Fallback 模块 — 三级兜底处理

## 模块概述

Fallback 模块在检索结果不足时提供分层兜底，确保用户始终获得有意义的回复。

- **补充检索**：放宽 top_k 重新检索，尝试从更多候选块中找到相关结果
- **联网搜索**：调用 DuckDuckGo 搜索引擎获取外部知识
- **诚实告知**：所有手段用尽后的最终兜底消息

## 文件结构

```
fallback/
├── __init__.py          # 导出 + get_fallback_handler 单例工厂
├── handler.py           # FallbackHandler — 兜底主编排器
├── supplementary.py     # SupplementaryRetriever — 补充检索
├── web_search.py        # WebSearcher — DuckDuckGo 联网搜索
└── README.md
```

## 快速开始

```python
from fallback import get_fallback_handler, reset_fallback_handler
from models.enums import RetrievalEval

handler = get_fallback_handler()

# NEED_MORE → 补充检索
ctx.retrieval_eval = RetrievalEval.NEED_MORE
ctx = await handler.handle(ctx, retrieval_layer)
# → ctx.retrieval_eval 可能变为 SUFFICIENT 或仍为 NEED_MORE
# → ctx.fallback_level = FallbackLevel.PARTIAL

# INSUFFICIENT → 联网搜索 → 诚实告知
ctx.retrieval_eval = RetrievalEval.INSUFFICIENT
ctx = await handler.handle(ctx)
# → 搜索成功: ctx.fallback_level = FallbackLevel.WEB_SEARCH
# → 搜索失败: ctx.fallback_level = FallbackLevel.NO_ANSWER

reset_fallback_handler()  # 测试用重置
```

## 兜底链路

```
检索评估结果
  │
  ├── SUFFICIENT → 不触发兜底，直接进入生成层
  │
  ├── NEED_MORE → 补充检索（放宽 top_k）
  │                ├── 找到足够结果 → 标记 PARTIAL，进入生成层
  │                └── 仍然不足   → 继续标记 NEED_MORE
  │
  └── INSUFFICIENT → 联网搜索（DuckDuckGo）
                       ├── 找到相关内容 → 标记 WEB_SEARCH，生成回答
                       └── 搜索失败     → 标记 NO_ANSWER，返回兜底消息
```

## 组件

### WebSearcher

```python
from fallback import WebSearcher

searcher = WebSearcher()
result = await searcher.search("Python RAG 框架")
# → 拼接的搜索结果文本，失败返回 None
```

### SupplementaryRetriever

```python
from fallback import SupplementaryRetriever

supp = SupplementaryRetriever()
ctx = await supp.retrieve(ctx, retrieval_layer)
# → 合并了放宽 top_k 后的新检索结果
```

## 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `web_search.enabled` | true | 联网搜索开关 |
| `web_search.provider` | duckduckgo | 搜索提供商 |
| `web_search.timeout_seconds` | 10 | 搜索超时 |
| `fallback.max_retrieval_rounds` | 2 | 补充检索最大轮次 |
| `fallback.no_answer_message` | "抱歉，当前知识库中..." | 诚实告知消息 |

## 依赖

```bash
pip install duckduckgo-search  # 联网搜索
```
