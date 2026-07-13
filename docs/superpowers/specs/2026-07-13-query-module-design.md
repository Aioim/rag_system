# Query 模块（查询理解层）— 设计文档

> 日期：2026-07-13
> 状态：待实现

---

## 1. 定位

`src/query/` 是 RAG Pipeline 的**查询理解层**，负责将原始用户 Query 转化为可用于检索的结构化上下文，填充 `PipelineContext`。

## 2. 模块结构

```
src/query/
├── __init__.py              # 导出 + 全局单例 query_layer
├── layer.py                 # QueryUnderstandingLayer 主编排器
├── intent_classifier.py     # 意图分类 + 清晰度判断（合并单次 LLM 调用）
├── context_fuser.py         # 多轮上下文融合（指代消解 + 追问补全）
└── rewriters/
    ├── __init__.py           # QueryRewriter 编排器
    ├── base.py               # BaseRewriter 抽象基类
    ├── hyde.py               # HyDE 假设答案改写
    ├── keyword.py            # 关键词提取（给 BM25）
    └── synonym.py            # 同义词扩展
```

## 3. 数据流

```
query (str) + session_id (str | None)
    │
    ▼
QueryUnderstandingLayer.process()
    │
    ├── 1. 别名映射 → resolve_aliases_in_text()
    │       └── 调用已有的 config.aliases 模块
    │
    ├── 2. 意图分类（+清晰度判断）→ IntentClassifier.classify()
    │       └── 单次 LLM 调用（Haiku, temp=0, JSON 输出）
    │       └── 模糊 → needs_clarification=True，短路返回
    │
    ├── 3. 多轮上下文融合 → ContextFuser.fuse()  [仅当 session_id 存在]
    │       ├── 取 SessionManager.get_context() → 历史消息 + 摘要
    │       ├── LLM 指代消解 + 追问补全
    │       └── 输出独立完整 query
    │
    └── 4. 查询改写（并行 3 路）→ QueryRewriter.rewrite()
            ├── HyDERewriter      → 假设答案文本
            ├── KeywordRewriter   → BM25 关键词
            └── SynonymRewriter   → 同义词变体查询
                    │
                    ▼
            PipelineContext（填充 intent, rewritten_queries,
                            needs_clarification, clarification_question,
                            session 等字段）
```

## 4. 组件设计

### 4.1 IntentClassifier — 意图分类 + 清晰度判断

**职责**：将用户 query 分类为 4 种意图（concept / procedure / compare / lookup），同时判断问题是否清晰。

**合并为单次调用的理由**：判断模糊度和判断意图在语义上天然关联（"帮帮我"既模糊又不属于任何意图），一次调用减少延迟。

**输入**：`query: str`
**输出**：`IntentResult(intent, is_clear, clarification_question)`

```python
@dataclass
class IntentResult:
    intent: Intent              # concept | procedure | compare | lookup
    is_clear: bool
    clarification_question: str | None

class IntentClassifier:
    def __init__(self, llm): ...
    async def classify(self, query: str) -> IntentResult: ...
```

**实现要点**：
- 使用轻量 LLM（Haiku），temperature=0
- Prompt 要求输出严格 JSON：`{"intent": "...", "is_clear": true/false, "clarification_question": null/string}`
- 失败降级：JSON 解析失败 → intent=concept, is_clear=True（兜底走正常检索）

### 4.2 ContextFuser — 多轮上下文融合

**职责**：将多轮对话中的追问/指代补全为独立完整问题。

**示例**：
- 第 1 轮："年假怎么申请？" → 第 2 轮："需要什么材料？" → 补全为"申请年假需要什么材料？"
- "它的有效期是多久？" → 补全为"VPN 远程接入的有效期是多久？"

```python
class ContextFuser:
    def __init__(self, llm, session_manager: SessionManager): ...
    async def fuse(self, query: str, session_id: str) -> str: ...
```

**实现要点**：
- 输入当前 query + 会话历史（最近 N 条消息 + 摘要）
- LLM 判断是否包含指代/省略 → 是则补全为完整问题，否则原样返回
- 使用轻量 LLM（Haiku），temperature=0

### 4.3 Rewriters — 查询改写

#### 统一接口

```python
class BaseRewriter(ABC):
    @abstractmethod
    async def rewrite(self, query: str) -> list[str]:
        """返回改写后的查询列表（可为空）"""
        ...
```

#### 4.3.1 HyDERewriter

- **原理**：先让 LLM 生成"假设答案"，用假设答案的 embedding 去检索（比原 query 在语义空间中更接近真实文档）
- **实现**：LLM 生成 1 段 100~200 字的假设答案文本
- **输出**：`["假设答案文本"]`

#### 4.3.2 KeywordRewriter

- **原理**：从 query 中提取纯关键词，给 BM25 做稀疏检索
- **实现**：轻量 LLM 或规则（jieba 分词 + TF-IDF/停用词过滤）提取关键词
- **输出**：`["关键词1 关键词2 关键词3"]`

#### 4.3.3 SynonymRewriter

- **原理**：用同义词/近义词扩展查询，增加召回覆盖面
- **实现**：维护同义词映射表 + LLM 生成近义变体
- **输出**：`["同义变体1", "同义变体2"]`

#### QueryRewriter 编排器

```python
class QueryRewriter:
    def __init__(self, llm):
        self.rewriters: list[BaseRewriter] = [
            HyDERewriter(llm),
            KeywordRewriter(llm),
            SynonymRewriter(llm),
        ]

    async def rewrite(self, query: str) -> list[str]:
        """并行执行所有 rewriter，去重合并结果"""
        results = await asyncio.gather(
            *(r.rewrite(query) for r in self.rewriters)
        )
        # 扁平化 + 去重 + 原始 query 始终在第一位
        all_queries = [query]
        for r in results:
            for q in r:
                if q not in all_queries:
                    all_queries.append(q)
        return all_queries
```

### 4.4 QueryUnderstandingLayer — 主编排器

```python
class QueryUnderstandingLayer:
    def __init__(self, llm, session_manager: SessionManager):
        self.intent_classifier = IntentClassifier(llm)
        self.context_fuser = ContextFuser(llm, session_manager)
        self.rewriter = QueryRewriter(llm)

    async def process(
        self,
        query: str,
        session_id: str | None = None,
        collection: str = "default",
    ) -> PipelineContext:
        ctx = PipelineContext(query=query, collection=collection)

        # 1. 别名映射
        query = resolve_aliases_in_text(query)

        # 2. 意图分类 + 清晰度判断
        result = await self.intent_classifier.classify(query)
        ctx.intent = result.intent
        if not result.is_clear:
            ctx.needs_clarification = True
            ctx.clarification_question = result.clarification_question
            return ctx  # 短路返回

        # 3. 多轮上下文融合
        if session_id:
            query = await self.context_fuser.fuse(query, session_id)
            ctx.query = query  # 更新为补全后的查询
            ctx.session = self.session_manager.get(session_id)

        # 4. 查询改写（并行）
        ctx.rewritten_queries = await self.rewriter.rewrite(query)
        return ctx
```

## 5. 依赖关系

| 模块 | 依赖 |
|------|------|
| `layer.py` | `config.aliases`, `session.manager`, `models.context` |
| `intent_classifier.py` | `models.enums` (Intent), LLM |
| `context_fuser.py` | `session.manager` (SessionManager) |
| `rewriters/*` | LLM, `models.enums` |

## 6. 全局单例

```python
# src/query/__init__.py
from query.layer import QueryUnderstandingLayer

query_layer: QueryUnderstandingLayer | None = None

def get_query_layer(llm, session_manager) -> QueryUnderstandingLayer:
    global query_layer
    if query_layer is None:
        query_layer = QueryUnderstandingLayer(llm, session_manager)
    return query_layer
```

## 7. 错误处理

| 场景 | 策略 |
|------|------|
| LLM 调用超时/失败 | 降级：intent=concept, is_clear=True，跳过改写，query 原样送入检索 |
| JSON 解析失败 | 降级：intent=concept, is_clear=True |
| Session 不存在 | 当作新会话处理，跳过融合步骤 |
| Rewriter 单个失败 | 不影响其他 rewriter，失败的返回空 list |

## 8. 不在本期范围

- 用户画像/个性化注入
- 子问题拆分（Multi-hop Decomposition）— 后续新增 `DecompositionRewriter`
- Query2Doc / Step-Back Prompting — 后续扩展
- 查询缓存
