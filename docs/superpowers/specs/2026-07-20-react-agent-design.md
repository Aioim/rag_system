# ReAct Agent 模式 — 设计文档

> 日期：2026-07-20
> 状态：设计完成，待评审
> 分支：master

---

## 1. 需求概述

在现有线性 RAG Pipeline 基础上增加 **ReAct（Reasoning + Acting）代理模式**，使 LLM 能够自主决定检索时机、检索内容、以及是否需要多轮检索，提升复杂问题的回答质量。

### 核心决策

| 维度 | 决策 |
|------|------|
| 与现有 Pipeline 关系 | **新增可选模式**（`mode="react"`），与线性模式并存 |
| Agent 工具集 | `search(query)` + `web_search(query)` |
| 停止条件 | LLM 自主判断 FINISH + 硬上限 max_iterations（默认 5）+ 连续重复检测 |
| 推理可见性 | 默认不可见，`show_reasoning=True` 时返回完整推理链 |

---

## 2. 模块结构

```
src/
├── agent/                        # 🆕 新增模块
│   ├── __init__.py               # 导出 + get_react_agent 单例工厂
│   ├── react_agent.py            # ReActAgent — 思考→行动→观察 循环
│   ├── tools.py                  # Tool 定义：SearchTool / WebSearchTool
│   └── README.md                 # 模块文档
├── core/
│   ├── pipeline.py               # 🔧 修改：增加 mode 参数 + ReAct 分支
│   └── ...
├── models/
│   ├── context.py                # 🔧 修改：增加 react_traces / mode 字段
│   └── ...
```

**设计原则**：
- `agent/` 是独立模块，不依赖 `core/`，只依赖 `retrieval/` 和 `fallback/`
- ReAct 模式**跳过 QueryUnderstanding**（Agent 自行推理和改写查询）
- Agent 失败时**自动降级**到线性 Pipeline

---

## 3. ReAct Agent 核心循环

```
用户 query
    │
    ├── 1. 别名映射（轻量预处理，保留）
    │
    ▼
┌─────────────────────────────────────────────┐
│              ReAct Agent 循环                 │
│                                              │
│   ┌──────────┐    ┌──────────┐    ┌───────┐ │
│   │ Thought  │───▶│  Action  │───▶│Observe│ │
│   │ (LLM思考)│◀───│ (工具调用)│    │(结果) │ │
│   └──────────┘    └──────────┘    └───────┘ │
│         │                              │     │
│         └──── 信息充分 → FINISH ───────┘     │
└─────────────────────────────────────────────┘
    │
    ├── 3. 检索自评（复用 Self-RAG evaluator）
    ├── 4. 生成最终答案（复用 GenerationLayer）
    └── 5. 会话记录
```

### 3.1 Prompt 结构

每次 LLM 调用使用以下 system prompt：

```
你是企业知识库问答助手，具备搜索内部知识库和联网信息的能力。

## 工具
- search(query): 搜索内部知识库，返回相关文档片段（含来源标识）
- web_search(query): 联网搜索公开信息，用于内部知识库资料不足时

## 输出格式
每次响应必须严格按以下格式（不要再输出其他内容）：

THOUGHT: <你的推理过程：当前信息是否足够回答用户问题？还需要搜索什么？>
ACTION: <search | web_search | FINISH>
QUERY: <搜索关键词，仅当 ACTION 为 search 或 web_search 时必须填写>

当检索到的信息已充分覆盖用户问题的所有方面时：
THOUGHT: <对检索结果的分析总结，说明为何可以回答>
ACTION: FINISH
```

### 3.2 停止条件

| 条件 | 行为 |
|------|------|
| LLM 返回 `ACTION: FINISH` | 退出循环，进入生成阶段 |
| 达到 `max_iterations`（默认 5） | 强制退出，用已累积的上下文生成 |
| 连续 2 轮返回相同 `ACTION + QUERY` | 退出，避免死循环 |
| LLM 调用异常 | 降级到线性 Pipeline |

### 3.3 Observation 注入

每轮工具调用结果以如下格式拼入下一轮对话历史：

```
OBSERVATION (search, 3 chunks, 245ms):
[来源: doc_001] ......
[来源: doc_003] ......
[来源: doc_007] ......
```

Agent 在多轮对话中能看到完整的 `THOUGHT → ACTION → OBSERVATION` 历史，支持基于之前结果的推理。

---

## 4. Tool 定义

```python
@dataclass
class ToolResult:
    """工具调用结果"""
    tool: str           # "search" | "web_search"
    query: str          # 实际执行的 query
    content: str        # 搜索结果文本（格式化后）
    chunk_count: int    # 返回的 chunk 数量
    elapsed_ms: float   # 耗时

class SearchTool:
    """search(query) — 封装 RetrievalLayer，走完整混合检索"""
    def __init__(self, retrieval_layer: RetrievalLayer): ...
    async def run(self, query: str, collection: str) -> ToolResult: ...

class WebSearchTool:
    """web_search(query) — 封装 WebSearcher"""
    def __init__(self, web_searcher: WebSearcher): ...
    async def run(self, query: str) -> ToolResult: ...
```

### 4.1 SearchTool 行为

- 构造 `PipelineContext(query=query, collection=collection)`
- `rewritten_queries` 仅包含原始 query（Agent 自己负责换角度搜索，不依赖 QueryRewriter）
- 走完整检索链：向量+BM25 → RRF 融合 → CrossEncoder 精排+MMR
- 将 `reranked` top_k chunks 格式化为 LLM 可读文本
- 失败返回空内容（不抛异常，Agent 可自行决定是否 web_search）

### 4.2 WebSearchTool 行为

- 直接复用 `WebSearcher.search(query)`
- 结果同样格式化为纯文本
- 失败返回空内容

---

## 5. RAGPipeline 集成

### 5.1 入口变更

```python
async def run(
    self,
    query: str,
    session_id: str | None = None,
    collection: str = "default",
    mode: str = "linear",          # 🆕 "linear" | "react"
    max_iterations: int = 5,       # 🆕 ReAct 最大轮次
    show_reasoning: bool = False,  # 🆕 是否返回推理链
) -> PipelineContext:
```

- `mode="linear"`：走现有单次检索→兜底→生成流程，**完全不变**
- `mode="react"`：走 ReAct Agent 循环

### 5.2 ReAct 分支流程

```
1. 别名映射 — 保留轻量预处理（resolve_aliases_in_text）
2. ReAct Agent 循环
   ├── 每轮: LLM 决策 → 解析 THOUGHT/ACTION/QUERY → 执行 Tool → Observation 注入
   ├── 累积 ctx.react_traces: list[ReActTrace]
   └── 循环退出 → ctx.reranked = 所有轮次搜索结果去重合并（按 rerank_score 排序）
3. 检索自评 → 同现有 evaluator（SUFFICIENT / NEED_MORE / INSUFFICIENT）
   └── 若 INSUFFICIENT 且 Agent 未调用过 web_search → 联网兜底
4. 生成 → 复用 GenerationLayer.generate(ctx)
   └── 上下文包括所有轮次累积的检索结果
5. 会话记录
```

### 5.3 流式输出

`ReActAgent` 提供两种调用方式：

```python
class ReActAgent:
    # 非流式：返回最终结果
    async def run(self, query: str, collection: str) -> AgentResult: ...

    # 流式：异步生成器，逐事件推送
    async def run_stream(self, query: str, collection: str) -> AsyncGenerator[SSEEvent]: ...
```

`RAGPipeline` 对应增加流式入口：

```python
async def run_stream(
    self,
    query: str,
    session_id: str | None = None,
    collection: str = "default",
    mode: str = "linear",
    max_iterations: int = 5,
    show_reasoning: bool = False,
) -> AsyncGenerator[SSEEvent]: ...
```

流式模式下 `stream=True` 时 `show_reasoning` 自动视为 `True`（否则无中间事件可推）。

### 5.4 异常降级

| 异常场景 | 降级策略 |
|----------|----------|
| LLM 调用连续失败 2 次 | 降级到 linear 模式重跑 |
| Tool 执行异常 | Observation 中注入错误信息，Agent 继续下一轮 |
| Agent 循环异常退出 | 用已累积的 reranked 结果进入生成 |

### 5.5 推理链暴露

```python
@dataclass
class ReActTrace:
    iteration: int
    thought: str
    action: str          # "search" | "web_search" | "finish"
    query: str | None    # 搜索 query（FINISH 时为 None）
    observation: str | None  # 工具返回结果（FINISH 时为 None）
    elapsed_ms: float
```

- `show_reasoning=False`（默认）：traces 仅写入日志，不进入 `ctx.metadata`
- `show_reasoning=True`：traces 写入 `ctx.metadata["react_traces"]`，API 响应中透出

---

## 6. 数据模型变更

### 6.1 PipelineContext 新增字段

```python
@dataclass
class PipelineContext:
    # ... 现有字段不变 ...
    
    # 🆕 ReAct 相关
    react_traces: list[ReActTrace] = field(default_factory=list)
    mode: str = "linear"
    max_iterations: int = 5
```

### 6.2 ReActTrace（新增）

```python
@dataclass
class ReActTrace:
    """ReAct Agent 单步推理记录"""
    iteration: int
    thought: str
    action: str
    query: str | None = None
    observation: str | None = None
    elapsed_ms: float = 0.0
```

---

## 7. 配置

在 `config/defaults.yaml` 新增：

```yaml
agent:
  max_iterations: 5            # ReAct 最大循环轮次
  search_top_k: 3              # 每轮 search 返回给 LLM 的 chunk 数
  max_observation_chars: 3000  # 每轮 Observation 最大字符数（截断）
  llm_temperature: 0.0         # Agent 决策 LLM 温度（需确定性）
  max_consecutive_duplicates: 2  # 连续重复 Action+Query 次数上限
```

---

## 8. API 影响

### 8.1 ChatRequest 变更

```python
class ChatRequest:
    # ... 现有字段不变 ...
    mode: str = "linear"            # 🆕 "linear" | "react"
    stream: bool = False            # 现有字段；ReAct 模式下同样生效
    max_iterations: int = 5         # 🆕 ReAct 最大轮次
    show_reasoning: bool = False    # 🆕 是否返回推理链（stream=true 时自动视为 true）
```

### 8.2 ChatResponse 变更（非流式）

```python
class ChatResponse:
    # ... 现有字段不变 ...
    react_traces: list[ReActTrace] | None = None  # 🆕 show_reasoning=true 时返回
```

### 8.3 SSE 流式事件协议

`stream=true` 时，ReAct 模式推送以下 SSE 事件：

```
event: react_start
data: {"mode": "react", "query": "..."}

event: thought
data: {"iteration": 1, "thought": "用户想了解RAG优化方向，我需要先搜索..."}

event: action
data: {"iteration": 1, "action": "search", "query": "RAG 检索增强生成 优化方向"}

event: observation
data: {"iteration": 1, "chunk_count": 3, "elapsed_ms": 245}

event: thought      ← 第 2 轮（如需要）
data: {"iteration": 2, "thought": "..."}

event: action
data: {"iteration": 2, "action": "search", "query": "..."}

event: observation
data: {"iteration": 2, "chunk_count": 2, "elapsed_ms": 180}

event: thought      ← 最终 FINISH
data: {"iteration": 3, "thought": "信息已充分，可以回答", "action": "finish"}

event: react_end
data: {"total_iterations": 3, "total_elapsed_ms": 890}

event: answer       ← 逐 token 推送（复用 GenerationLayer 流式能力）
data: {"token": "RAG"}

event: answer
data: {"token": "架构"}

...

event: done
data: {"answer": "...", "sources": [...], "confidence": 0.85}
```

**事件说明**：

| 事件 | 触发时机 | 说明 |
|------|----------|------|
| `react_start` | Agent 循环开始 | 标识进入 ReAct 模式 |
| `thought` | LLM 返回思考结果 | 包含 reasoning 文本；FINISH 时 `action="finish"` |
| `action` | 解析到工具调用 | 即将执行 search/web_search |
| `observation` | 工具执行完成 | 包含 chunk_count 和耗时，不传具体内容（节省带宽） |
| `react_end` | Agent 循环结束 | 总轮次和总耗时汇总 |
| `answer` | LLM 生成中 | 逐 token 推送最终答案 |
| `done` | 全流程结束 | 最终结果（sources、confidence） |

**降级处理**：SSE 连接断开时 Agent 继续执行完毕，结果写入会话历史供后续查询。`show_reasoning=false` 时跳过 `thought/action/observation` 事件，只推送 `react_start → (等待) → react_end → answer... → done`。`mode="linear"` 的流式行为与现有保持一致（不推送 react_* 事件）。

---

## 9. 测试策略

| 层级 | 内容 |
|------|------|
| 单元测试 | `ReActAgent` 循环逻辑（mock LLM/Tool）、输出解析（THOUGHT/ACTION/QUERY 正则提取）、停止条件（max_iterations/重复检测/降级） |
| 集成测试 | `RAGPipeline.run(mode="react")` 端到端（mock LLM + 真实检索层）、降级到 linear 模式 |
| 边界测试 | 空 Tool 结果、LLM 格式错误输出、连续重复检测、单轮 FINISH |

Mock LLM 按 `iteration` 返回预设响应（参考现有 `QueryUnderstandingLayer` 自测的 mock 模式），不依赖真实 LLM。

---

## 10. 风险与限制

| 风险 | 缓解措施 |
|------|----------|
| ReAct 多轮调用增加延迟 | 简单问题走 linear；max_iterations 限制；Haiku 做 Agent 决策 |
| LLM 格式输出不稳定 | 正则容错解析 + 格式错误时记录日志并尝试修复 |
| Token 消耗增加 | Observation 截断（max_observation_chars）；累积上下文去重 |
| 与 linear 模式行为不一致 | 共享 GenerationLayer，确保最终答案风格一致 |
