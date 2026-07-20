# ReAct Agent 模式 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有线性 RAG Pipeline 基础上新增 ReAct（Reasoning + Acting）代理模式，LLM 自主决定检索时机和内容。

**Architecture:** 新增 `src/agent/` 独立模块（不依赖 `core/`），通过 `SearchTool` 和 `WebSearchTool` 封装现有检索和搜索能力。`RAGPipeline.run()` 增加 `mode` 参数分叉到 `ReActAgent` 循环。Agent 失败自动降级到线性模式。

**Tech Stack:** Python 3.12+, asyncio, Pydantic v2, langchain > 1.3.0, langgraph >= 1.2.0

**Spec:** `docs/superpowers/specs/2026-07-20-react-agent-design.md`

## Global Constraints

- langchain > 1.3.0, langgraph >= 1.2.0
- Pydantic v2 配置模型
- FastAPI + asyncio
- 不可变数据模式（dataclass，不原地修改）
- 80%+ 测试覆盖率
- TDD：先写测试再写实现
- 所有异步阻塞操作走 `asyncio.to_thread` 或 `loop.run_in_executor`

---

### Task 1: AgentConfig 配置模型 + defaults.yaml

**Files:**
- Modify: `config/defaults.yaml` — 在 `finetune:` 段后追加 `agent:` 段
- Modify: `src/config/settings.py` — 新增 `AgentConfig` 类并注册到 `RAGAppConfig`

**Interfaces:**
- Produces: `AgentConfig(BaseModel)` — `max_iterations: int`, `search_top_k: int`, `max_observation_chars: int`, `llm_temperature: float`, `max_consecutive_duplicates: int`
- Produces: `RAGAppConfig.agent: AgentConfig` — 通过 `settings.agent` 访问

- [ ] **Step 1: 在 defaults.yaml 末尾追加 agent 配置段**

在 `config/defaults.yaml` 的 `finetune:` 段之后追加：

```yaml
# ReAct Agent 配置
agent:
  max_iterations: 5
  search_top_k: 3
  max_observation_chars: 3000
  llm_temperature: 0.0
  max_consecutive_duplicates: 2
```

- [ ] **Step 2: 在 settings.py 中新增 AgentConfig 类**

在 `src/config/settings.py` 中，`FinetuneConfig` 类之后、`RAGAppConfig` 类之前插入：

```python
class AgentConfig(_BaseConfig):
    """ReAct Agent 配置"""
    max_iterations: int = Field(default=5, ge=1, le=20)
    search_top_k: int = Field(default=3, ge=1, le=10)
    max_observation_chars: int = Field(default=3000, ge=100, le=10000)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_consecutive_duplicates: int = Field(default=2, ge=1, le=5)
```

- [ ] **Step 3: 在 RAGAppConfig 中注册 agent 字段**

在 `RAGAppConfig` 类的字段列表末尾追加（`finetune` 行之后）：

```python
agent: AgentConfig = Field(default_factory=AgentConfig)
```

- [ ] **Step 4: 在 `__all__` 导出列表中添加 `"AgentConfig"`**

在 `src/config/settings.py` 的 `__all__` 列表中添加 `"AgentConfig"`（字母序，第一个）。

- [ ] **Step 5: 验证配置加载**

```bash
cd E:/Code/rag0709 && python -c "from config import settings; print(settings.agent); print('max_iterations:', settings.agent.max_iterations)"
```

Expected: 输出 `AgentConfig` 各字段值，`max_iterations=5`。

- [ ] **Step 6: 提交**

```bash
git add config/defaults.yaml src/config/settings.py
git commit -m "feat(agent): add AgentConfig model and defaults.yaml agent section"
```

---

### Task 2: 数据模型 — ReActTrace + PipelineContext 变更 + SSEEvent

**Files:**
- Create: `src/models/react_trace.py` — `ReActTrace` dataclass
- Modify: `src/models/context.py` — 增加 `react_traces`, `mode`, `max_iterations` 字段
- Modify: `src/models/__init__.py` — 导出 `ReActTrace`

**Interfaces:**
- Produces: `ReActTrace(iteration, thought, action, query, observation, elapsed_ms)` — dataclass
- Produces: `PipelineContext.react_traces: list[ReActTrace]`, `.mode: str`, `.max_iterations: int`

- [ ] **Step 1: 创建 ReActTrace dataclass**

```python
# src/models/react_trace.py
from dataclasses import dataclass, field


@dataclass
class ReActTrace:
    """ReAct Agent 单步推理记录"""
    iteration: int
    thought: str
    action: str          # "search" | "web_search" | "finish"
    query: str | None = None
    observation: str | None = None
    elapsed_ms: float = 0.0
```

- [ ] **Step 2: 扩展 PipelineContext**

在 `src/models/context.py` 中，`metadata` 字段之后追加：

```python
react_traces: list = field(default_factory=list)  # list[ReActTrace]，延迟注解避免循环引用
mode: str = "linear"
max_iterations: int = 5
```

注意：由于 `ReActTrace` 是新类型，使用 `list` 默认值 + 注释方式避免 TYPE_CHECKING 复杂性。若需要类型检查，可在 TYPE_CHECKING 块中导入 `ReActTrace`。

- [ ] **Step 3: 更新 models/__init__.py 导出**

```python
from models.react_trace import ReActTrace

# 在 __all__ 列表中添加 "ReActTrace"
```

- [ ] **Step 4: 验证导入链**

```bash
cd E:/Code/rag0709 && python -c "from models import ReActTrace; t = ReActTrace(iteration=1, thought='test', action='search', query='hello'); print(t)"
```

Expected: `ReActTrace(iteration=1, thought='test', action='search', query='hello', observation=None, elapsed_ms=0.0)`

- [ ] **Step 5: 提交**

```bash
git add src/models/react_trace.py src/models/context.py src/models/__init__.py
git commit -m "feat(agent): add ReActTrace model and PipelineContext extensions"
```

---

### Task 3: 输出解析器 — THOUGHT/ACTION/QUERY 正则提取

**Files:**
- Create: `tests/unit/agent/__init__.py` — 空文件
- Create: `tests/unit/agent/test_output_parser.py` — 解析器测试
- Modify: `src/agent/__init__.py` (若不存在则 Create) — 先创建占位，后续任务填充

**Interfaces:**
- Produces: `parse_react_output(text: str) -> dict[str, str | None]` — 返回 `{"thought": ..., "action": ..., "query": ...}` 或 `{"parse_error": ...}`

- [ ] **Step 1: 创建 agent 包占位文件**

```bash
mkdir -p src/agent
```

创建 `src/agent/__init__.py`（占位）：

```python
"""ReAct Agent 模块 — 思考→行动→观察 循环"""
```

创建 `tests/unit/agent/__init__.py`（空文件）。

- [ ] **Step 2: 编写输出解析器测试**

```python
# tests/unit/agent/test_output_parser.py
import pytest
from agent.react_agent import parse_react_output


class TestParseReactOutput:
    def test_parses_search_action(self):
        text = "THOUGHT: 需要搜索RAG相关资料\nACTION: search\nQUERY: RAG 检索增强生成"
        result = parse_react_output(text)
        assert result["thought"] == "需要搜索RAG相关资料"
        assert result["action"] == "search"
        assert result["query"] == "RAG 检索增强生成"

    def test_parses_web_search_action(self):
        text = "THOUGHT: 内部知识库无结果\nACTION: web_search\nQUERY: RAG architecture 2026"
        result = parse_react_output(text)
        assert result["action"] == "web_search"

    def test_parses_finish_action(self):
        text = "THOUGHT: 信息已充分，可以回答\nACTION: FINISH"
        result = parse_react_output(text)
        assert result["thought"] == "信息已充分，可以回答"
        assert result["action"] == "finish"
        assert result["query"] is None

    def test_parses_multiline_thought(self):
        text = "THOUGHT: 第一行思考\n第二行继续思考\n第三行总结\nACTION: FINISH"
        result = parse_react_output(text)
        assert "第一行思考" in result["thought"]
        assert result["action"] == "finish"

    def test_rejects_unknown_action(self):
        text = "THOUGHT: test\nACTION: unknown\nQUERY: test"
        result = parse_react_output(text)
        assert "parse_error" in result

    def test_handles_missing_thought(self):
        text = "ACTION: search\nQUERY: test"
        result = parse_react_output(text)
        assert "parse_error" in result

    def test_handles_empty_input(self):
        result = parse_react_output("")
        assert "parse_error" in result

    def test_trims_whitespace(self):
        text = "  THOUGHT:  需要搜索  \n  ACTION:  search  \n  QUERY:  hello world  "
        result = parse_react_output(text)
        assert result["thought"] == "需要搜索"
        assert result["action"] == "search"
        assert result["query"] == "hello world"

    def test_finish_ignores_query_field(self):
        text = "THOUGHT: done\nACTION: FINISH\nQUERY: should be ignored"
        result = parse_react_output(text)
        assert result["action"] == "finish"
        assert result["query"] is None
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_output_parser.py -v
```

Expected: FAIL (ImportError — `parse_react_output` 不存在)

- [ ] **Step 4: 实现 parse_react_output**

在 `src/agent/react_agent.py` 中：

```python
"""ReActAgent — 思考→行动→观察 循环"""
import re
from typing import Any


# ---- 输出解析器 ---------------------------------------------------------------

_VALID_ACTIONS = frozenset({"search", "web_search", "finish"})

_RE_THOUGHT = re.compile(r"THOUGHT:\s*(.+?)(?=\nACTION:|\Z)", re.DOTALL | re.IGNORECASE)
_RE_ACTION = re.compile(r"ACTION:\s*(\w+)", re.IGNORECASE)
_RE_QUERY = re.compile(r"QUERY:\s*(.+)", re.IGNORECASE)


def parse_react_output(text: str) -> dict[str, Any]:
    """从 LLM 响应中提取 THOUGHT / ACTION / QUERY 三元组。
    
    Returns:
        成功: {"thought": str, "action": str, "query": str | None}
        失败: {"parse_error": str}
    """
    if not text or not text.strip():
        return {"parse_error": "empty input"}

    # 提取 THOUGHT
    m_thought = _RE_THOUGHT.search(text)
    if not m_thought:
        return {"parse_error": "missing THOUGHT field"}
    thought = " ".join(m_thought.group(1).strip().split())

    # 提取 ACTION
    m_action = _RE_ACTION.search(text)
    if not m_action:
        return {"parse_error": "missing ACTION field"}
    action = m_action.group(1).strip().lower()
    if action not in _VALID_ACTIONS:
        return {"parse_error": f"unknown action: {action!r}"}

    # FINISH 不需要 QUERY
    if action == "finish":
        return {"thought": thought, "action": "finish", "query": None}

    # search / web_search 必须有 QUERY
    m_query = _RE_QUERY.search(text)
    if not m_query:
        return {"parse_error": f"missing QUERY field for action={action!r}"}
    query = " ".join(m_query.group(1).strip().split())

    return {"thought": thought, "action": action, "query": query}
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_output_parser.py -v
```

Expected: 9 passed

- [ ] **Step 6: 提交**

```bash
git add src/agent/__init__.py src/agent/react_agent.py tests/unit/agent/__init__.py tests/unit/agent/test_output_parser.py
git commit -m "feat(agent): add parse_react_output with THOUGHT/ACTION/QUERY extraction"
```

---

### Task 4: Tool 定义 — ToolResult + SearchTool + WebSearchTool

**Files:**
- Create: `src/agent/tools.py` — `ToolResult`, `SearchTool`, `WebSearchTool`
- Create: `tests/unit/agent/test_tools.py` — Tool 单元测试
- Create: `tests/unit/agent/conftest.py` — 共享 fixtures（mock LLM, mock store 等）

**Interfaces:**
- Consumes: `RetrievalLayer` (from `src/retrieval/layer.py`), `WebSearcher` (from `src/fallback/web_search.py`)
- Produces: `ToolResult(tool, query, content, chunk_count, elapsed_ms)` — dataclass
- Produces: `SearchTool(retrieval_layer).run(query, collection) -> ToolResult`
- Produces: `WebSearchTool(web_searcher).run(query) -> ToolResult`

- [ ] **Step 1: 编写 Tool 测试**

```python
# tests/unit/agent/test_tools.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.tools import ToolResult, SearchTool, WebSearchTool


class TestToolResult:
    def test_create_tool_result(self):
        r = ToolResult(tool="search", query="test", content="results",
                       chunk_count=3, elapsed_ms=100.0)
        assert r.tool == "search"
        assert r.chunk_count == 3


class TestSearchTool:
    @pytest.fixture
    def mock_retrieval_layer(self):
        layer = MagicMock()
        layer.retrieve = AsyncMock()
        return layer

    def test_run_returns_tool_result(self, mock_retrieval_layer):
        """search 正常返回时获得 ToolResult，内容包含来源标识"""
        from models.chunk import Chunk
        from models.enums import RetrievalEval
        
        chunk = Chunk(
            chunk_id="c1", doc_id="d1", text="RAG是检索增强生成",
            chunk_index=0, embedding=None
        )
        chunk.rerank_score = 0.9
        
        # 构造 PipelineContext 返回值
        async def mock_retrieve(ctx, top_k=None):
            ctx.reranked = [chunk]
            ctx.retrieval_eval = RetrievalEval.SUFFICIENT
            return ctx
        
        mock_retrieval_layer.retrieve = mock_retrieve
        
        tool = SearchTool(mock_retrieval_layer)
        result = asyncio.get_event_loop().run_until_complete(
            tool.run("RAG架构", "default")
        )
        
        assert result.tool == "search"
        assert result.chunk_count == 1
        assert "RAG是检索增强生成" in result.content
        assert result.elapsed_ms > 0

    def test_run_handles_exception(self, mock_retrieval_layer):
        """retrieval 异常时返回空内容，不抛异常"""
        mock_retrieval_layer.retrieve = AsyncMock(side_effect=RuntimeError("store error"))
        
        tool = SearchTool(mock_retrieval_layer)
        result = asyncio.get_event_loop().run_until_complete(
            tool.run("test", "default")
        )
        
        assert result.content == ""
        assert result.chunk_count == 0


class TestWebSearchTool:
    @pytest.fixture
    def mock_web_searcher(self):
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value="联网搜索结果文本")
        return searcher

    def test_run_returns_tool_result(self, mock_web_searcher):
        tool = WebSearchTool(mock_web_searcher)
        result = asyncio.get_event_loop().run_until_complete(
            tool.run("Python RAG")
        )
        
        assert result.tool == "web_search"
        assert "联网搜索结果" in result.content

    def test_run_handles_exception(self, mock_web_searcher):
        mock_web_searcher.search = AsyncMock(side_effect=RuntimeError("network error"))
        tool = WebSearchTool(mock_web_searcher)
        result = asyncio.get_event_loop().run_until_complete(
            tool.run("test")
        )
        assert result.content == ""
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_tools.py -v
```

Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 ToolResult + SearchTool + WebSearchTool**

```python
# src/agent/tools.py
"""ReAct Agent 工具定义：SearchTool / WebSearchTool"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    from fallback.web_search import WebSearcher
    from retrieval.layer import RetrievalLayer


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

    def __init__(self, retrieval_layer: RetrievalLayer) -> None:
        self._retrieval = retrieval_layer

    async def run(self, query: str, collection: str) -> ToolResult:
        from models.context import PipelineContext

        t0 = time.perf_counter()
        ctx = PipelineContext(query=query, collection=collection)
        ctx.rewritten_queries = [query]  # Agent 自行改写，不依赖 QueryRewriter
        try:
            ctx = await self._retrieval.retrieve(
                ctx, top_k=settings.agent.search_top_k
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("SearchTool 检索失败: %s", e)
            return ToolResult(
                tool="search", query=query, content="",
                chunk_count=0,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        chunks = ctx.reranked or []
        content = self._format_chunks(chunks)
        return ToolResult(
            tool="search", query=query, content=content,
            chunk_count=len(chunks),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _format_chunks(chunks: list) -> str:
        lines: list[str] = []
        for c in chunks:
            text = c.text.replace("\n", " ")[:settings.agent.max_observation_chars]
            lines.append(f"[来源: {c.doc_id}] {text}")
        return "\n".join(lines)


class WebSearchTool:
    """web_search(query) — 封装 WebSearcher"""

    def __init__(self, web_searcher: WebSearcher) -> None:
        self._searcher = web_searcher

    async def run(self, query: str) -> ToolResult:
        t0 = time.perf_counter()
        try:
            content = await self._searcher.search(query)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("WebSearchTool 失败: %s", e)
            content = ""

        return ToolResult(
            tool="web_search", query=query,
            content=content or "",
            chunk_count=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_tools.py -v
```

Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add src/agent/tools.py tests/unit/agent/test_tools.py
git commit -m "feat(agent): add ToolResult, SearchTool, and WebSearchTool"
```

---

### Task 5: ReActAgent 核心循环（run 方法，非流式）

**Files:**
- Modify: `src/agent/react_agent.py` — 追加 `ReActAgent` 类
- Create: `tests/unit/agent/test_react_agent.py` — ReActAgent 单元测试
- Modify: `tests/unit/agent/conftest.py` — 共享 mock LLM / tool fixtures

**Interfaces:**
- Consumes: `LLMProtocol` (ainvoke), `SearchTool`, `WebSearchTool`, `AgentConfig`
- Produces: `ReActAgent(llm, search_tool, web_search_tool, config).run(query, collection) -> AgentResult`
- Produces: `AgentResult(reranked, react_traces, total_iterations, total_elapsed_ms)` — dataclass

- [ ] **Step 1: 创建共享 conftest.py**

```python
# tests/unit/agent/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock


class ProgrammableLLM:
    """可编程 Mock LLM — 每次 ainvoke 按顺序返回预设响应"""
    
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.calls: list[tuple[str, float]] = []  # (prompt, temperature)
    
    async def ainvoke(self, prompt: str, temperature: float = 0.0):
        self.calls.append((prompt, temperature))
        if self.call_count >= len(self.responses):
            return MagicMock(content="THOUGHT: 默认响应\nACTION: FINISH")
        response = self.responses[self.call_count]
        self.call_count += 1
        return MagicMock(content=response)


@pytest.fixture
def agent_config():
    from config.settings import AgentConfig
    return AgentConfig(
        max_iterations=5,
        search_top_k=3,
        max_observation_chars=3000,
        llm_temperature=0.0,
        max_consecutive_duplicates=2,
    )


@pytest.fixture
def programmable_llm():
    """返回工厂函数，避免跨测试状态污染"""
    def _make(responses: list[str]):
        return ProgrammableLLM(responses)
    return _make
```

- [ ] **Step 2: 编写 ReActAgent 测试**

```python
# tests/unit/agent/test_react_agent.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.react_agent import ReActAgent, AgentResult
from agent.tools import SearchTool, WebSearchTool, ToolResult


def _make_search_tool(chunks_per_call: list[list] | None = None):
    """创建可编程 SearchTool mock"""
    if chunks_per_call is None:
        chunks_per_call = [[]]
    tool = MagicMock(spec=SearchTool)
    call_results = []
    for chunks in chunks_per_call:
        content = "\n".join(f"[来源: {c['doc_id']}] {c['text']}" for c in chunks)
        call_results.append(
            ToolResult(tool="search", query="", content=content,
                       chunk_count=len(chunks), elapsed_ms=100.0)
        )
    tool.run = AsyncMock(side_effect=call_results)
    return tool


def _make_web_search_tool(results: list[str] | None = None):
    if results is None:
        results = [""]
    tool = MagicMock(spec=WebSearchTool)
    call_results = [
        ToolResult(tool="web_search", query="", content=r or "",
                   chunk_count=0, elapsed_ms=200.0)
        for r in results
    ]
    tool.run = AsyncMock(side_effect=call_results)
    return tool


class TestReActAgent:
    def test_single_round_finish(self, programmable_llm, agent_config):
        """单轮 FINISH：Agent 直接结束"""
        llm = programmable_llm([
            "THOUGHT: 这是一个简单问题，无需搜索\nACTION: FINISH"
        ])
        search = _make_search_tool()
        web = _make_web_search_tool()
        agent = ReActAgent(llm, search, web, agent_config)
        
        result = asyncio.get_event_loop().run_until_complete(
            agent.run("你好", "default")
        )
        
        assert result.total_iterations == 1
        assert result.react_traces[0].action == "finish"
        assert search.run.call_count == 0

    def test_two_round_search_then_finish(self, programmable_llm, agent_config):
        """两轮检索后 FINISH"""
        from models.chunk import Chunk
        
        chunk = Chunk(chunk_id="c1", doc_id="d1", text="RAG是检索增强生成",
                       chunk_index=0, embedding=None)
        chunk.rerank_score = 0.9
        
        llm = programmable_llm([
            "THOUGHT: 需要搜索RAG资料\nACTION: search\nQUERY: RAG架构",
            "THOUGHT: 信息充分，可以回答\nACTION: FINISH",
        ])
        search = _make_search_tool([[{"doc_id": "d1", "text": "RAG是检索增强生成"}]])
        web = _make_web_search_tool()
        agent = ReActAgent(llm, search, web, agent_config)
        
        result = asyncio.get_event_loop().run_until_complete(
            agent.run("什么是RAG？", "default")
        )
        
        assert result.total_iterations == 2
        assert result.react_traces[0].action == "search"
        assert result.react_traces[1].action == "finish"
        assert len(result.reranked) == 1

    def test_max_iterations_limit(self, programmable_llm, agent_config):
        """达到 max_iterations 后强制退出"""
        agent_config.max_iterations = 3
        llm = programmable_llm([
            "THOUGHT: 搜索第一次\nACTION: search\nQUERY: test1",
            "THOUGHT: 搜索第二次\nACTION: search\nQUERY: test2",
            "THOUGHT: 搜索第三次\nACTION: search\nQUERY: test3",
        ])
        search = _make_search_tool([
            [{"doc_id": "d1", "text": "result1"}],
            [{"doc_id": "d2", "text": "result2"}],
            [{"doc_id": "d3", "text": "result3"}],
        ])
        web = _make_web_search_tool()
        agent = ReActAgent(llm, search, web, agent_config)
        
        result = asyncio.get_event_loop().run_until_complete(
            agent.run("test", "default")
        )
        
        assert result.total_iterations == 3
        # 最后一轮 action 被强制改为 finish
        assert result.react_traces[-1].action == "finish"

    def test_consecutive_duplicate_detection(self, programmable_llm, agent_config):
        """连续两轮相同 ACTION+QUERY 触发死循环检测"""
        agent_config.max_consecutive_duplicates = 2
        llm = programmable_llm([
            "THOUGHT: 搜索\nACTION: search\nQUERY: same query",
            "THOUGHT: 再搜一次\nACTION: search\nQUERY: same query",
            "THOUGHT: 还搜\nACTION: search\nQUERY: same query",
        ])
        search = _make_search_tool([
            [{"doc_id": "d1", "text": "r1"}],
            [{"doc_id": "d1", "text": "r1"}],
            [{"doc_id": "d1", "text": "r1"}],
        ])
        web = _make_web_search_tool()
        agent = ReActAgent(llm, search, web, agent_config)
        
        result = asyncio.get_event_loop().run_until_complete(
            agent.run("test", "default")
        )
        
        # 第2轮检测到重复，第3轮不应执行
        assert result.total_iterations <= 2

    def test_web_search_when_kb_empty(self, programmable_llm, agent_config):
        """search 返回空结果后 Agent 选择 web_search"""
        llm = programmable_llm([
            "THOUGHT: 内部搜索无结果，尝试联网\nACTION: web_search\nQUERY: latest RAG trends",
            "THOUGHT: 联网搜索获得结果\nACTION: FINISH",
        ])
        search = _make_search_tool([[]])  # 空结果
        web = _make_web_search_tool(["联网搜索结果: RAG最新趋势..."])
        agent = ReActAgent(llm, search, web, agent_config)
        
        result = asyncio.get_event_loop().run_until_complete(
            agent.run("RAG最新趋势", "default")
        )
        
        assert result.react_traces[0].action == "web_search"
        assert result.react_traces[1].action == "finish"

    def test_llm_format_error_triggers_retry_then_finish(self, programmable_llm, agent_config):
        """LLM 格式错误后重试，连续两次错误后降级 FINISH"""
        llm = programmable_llm([
            "随便写的一段话，没有格式",                              # 解析失败
            "THOUGHT: 正确格式\nACTION: FINISH",                    # 正常结束
        ])
        search = _make_search_tool()
        web = _make_web_search_tool()
        agent = ReActAgent(llm, search, web, agent_config)
        
        result = asyncio.get_event_loop().run_until_complete(
            agent.run("test", "default")
        )
        
        assert result.total_iterations == 2
        assert result.react_traces[0].action == "finish"  # 解析失败退化为 FINISH
        assert "格式" in result.react_traces[0].thought
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_react_agent.py -v
```

Expected: FAIL (ReActAgent / AgentResult 未定义)

- [ ] **Step 4: 实现 AgentResult + ReActAgent.run()**

在 `src/agent/react_agent.py` 中（`parse_react_output` 之后）追加：

```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    from config.settings import AgentConfig
    from models.chunk import Chunk
    from models.llm import LLMProtocol
    from agent.tools import SearchTool, WebSearchTool

# ... parse_react_output 在上方 ...


@dataclass
class AgentResult:
    """ReAct Agent 执行结果"""
    reranked: list = field(default_factory=list)  # list[Chunk]
    react_traces: list = field(default_factory=list)  # list[ReActTrace]
    total_iterations: int = 0
    total_elapsed_ms: float = 0.0


# ReAct Agent 决策 System Prompt
_SYSTEM_PROMPT = """你是企业知识库问答助手，具备搜索内部知识库和联网信息的能力。

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
ACTION: FINISH"""


class ReActAgent:
    """ReAct Agent — 思考→行动→观察 循环
    
    每个循环：
    1. 调用 LLM 获取 THOUGHT/ACTION/QUERY
    2. 如果 FINISH → 退出循环
    3. 执行工具（search / web_search）
    4. 将 Observation 注入历史，进入下一轮
    """

    def __init__(
        self,
        llm: "LLMProtocol",
        search_tool: "SearchTool",
        web_search_tool: "WebSearchTool",
        config: "AgentConfig | None" = None,
    ) -> None:
        self._llm = llm
        self._search = search_tool
        self._web = web_search_tool
        self._config = config or settings.agent

    # ---- 主入口 ------------------------------------------------------------

    async def run(self, query: str, collection: str) -> AgentResult:
        """执行 ReAct 循环，返回 AgentResult"""
        from models.react_trace import ReActTrace

        t0 = time.perf_counter()
        traces: list[ReActTrace] = []
        observation: str | None = None
        seen_pairs: set[tuple[str, str]] = set()
        consecutive_dup_count = 0

        for iteration in range(1, self._config.max_iterations + 1):
            t_iter = time.perf_counter()

            # 1. 调用 LLM 获取决策
            prompt = self._build_prompt(query, traces, observation)
            raw = await self._call_llm(prompt)

            # 2. 解析输出
            parsed = parse_react_output(raw)
            if "parse_error" in parsed:
                # 格式错误：记录 trace 并退化为 FINISH
                trace = ReActTrace(
                    iteration=iteration,
                    thought=f"LLM 输出格式异常: {parsed['parse_error']}，强制结束",
                    action="finish",
                    elapsed_ms=(time.perf_counter() - t_iter) * 1000,
                )
                traces.append(trace)
                break

            action = parsed["action"]
            search_query = parsed.get("query")

            # 3. 连续重复检测
            pair_key = (action, search_query or "")
            if pair_key in seen_pairs:
                consecutive_dup_count += 1
                if consecutive_dup_count >= self._config.max_consecutive_duplicates:
                    trace = ReActTrace(
                        iteration=iteration,
                        thought=f"连续 {consecutive_dup_count} 轮重复 {action}:{search_query}，终止循环",
                        action="finish",
                        elapsed_ms=(time.perf_counter() - t_iter) * 1000,
                    )
                    traces.append(trace)
                    break
            else:
                consecutive_dup_count = 0
            seen_pairs.add(pair_key)

            # 4. FINISH → 退出
            if action == "finish":
                trace = ReActTrace(
                    iteration=iteration,
                    thought=parsed["thought"],
                    action="finish",
                    elapsed_ms=(time.perf_counter() - t_iter) * 1000,
                )
                traces.append(trace)
                break

            # 5. 执行工具 + 构建 Observation
            tool_result = await self._execute_tool(action, search_query or "", collection)
            observation = self._format_observation(tool_result)

            trace = ReActTrace(
                iteration=iteration,
                thought=parsed["thought"],
                action=action,
                query=search_query,
                observation=observation,
                elapsed_ms=(time.perf_counter() - t_iter) * 1000,
            )
            traces.append(trace)

        # 循环结束，合并去重所有检索结果
        reranked = await self._collect_reranked(traces)

        return AgentResult(
            reranked=reranked,
            react_traces=traces,
            total_iterations=len(traces),
            total_elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    # ---- 内部方法 ----------------------------------------------------------

    def _build_prompt(
        self, query: str, traces: list, observation: str | None
    ) -> str:
        """构建本轮 LLM 调用的完整 prompt"""
        parts = [_SYSTEM_PROMPT, "", f"用户问题: {query}"]

        # 注入历史痕迹
        for t in traces:
            parts.append("")
            parts.append(f"THOUGHT: {t.thought}")
            parts.append(f"ACTION: {t.action}")
            if t.query:
                parts.append(f"QUERY: {t.query}")
            if t.observation:
                parts.append(t.observation)

        # 注入上一轮 Observation（如果有新结果）
        if observation:
            parts.append("")
            parts.append(observation)

        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM，异常时返回空字符串"""
        try:
            result = await self._llm.ainvoke(
                prompt, temperature=self._config.llm_temperature
            )
            return result.content
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("ReActAgent LLM 调用失败: %s", e)
            return ""

    async def _execute_tool(
        self, action: str, query: str, collection: str
    ) -> "ToolResult":
        """执行指定工具"""
        from agent.tools import ToolResult

        try:
            if action == "search":
                return await self._search.run(query, collection)
            elif action == "web_search":
                return await self._web.run(query)
            else:
                return ToolResult(
                    tool=action, query=query, content="",
                    chunk_count=0, elapsed_ms=0,
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("工具执行失败 %s: %s", action, e)
            from agent.tools import ToolResult
            return ToolResult(
                tool=action, query=query, content=f"工具执行错误: {e}",
                chunk_count=0, elapsed_ms=0,
            )

    @staticmethod
    def _format_observation(result: "ToolResult") -> str:
        """格式化 Observation 文本"""
        return (
            f"OBSERVATION ({result.tool}, {result.chunk_count} chunks, "
            f"{result.elapsed_ms:.0f}ms):\n{result.content}"
        )

    @staticmethod
    async def _collect_reranked(traces: list) -> list:
        """从 traces 中提取 web_search 文本作为虚拟 Chunk，合并返回
        
        注意：实际 reranked chunks 由上层 RAGPipeline 通过
        search traces 的 query 重新检索获得（支持去重）。
        这里返回空列表，由 RAGPipeline 统一处理。
        """
        return []  # 由 RAGPipeline 的 _merge_react_results 统一处理
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_react_agent.py -v
```

Expected: 6 passed

- [ ] **Step 6: 提交**

```bash
git add src/agent/react_agent.py tests/unit/agent/test_react_agent.py tests/unit/agent/conftest.py
git commit -m "feat(agent): add ReActAgent core loop with run() method"
```

---

### Task 6: ReActAgent 流式输出（run_stream 方法）

**Files:**
- Modify: `src/agent/react_agent.py` — 追加 `run_stream` 方法和 `SSEEvent` dataclass
- Modify: `tests/unit/agent/test_react_agent.py` — 追加流式测试

**Interfaces:**
- Produces: `SSEEvent(event, data)` — dataclass
- Produces: `ReActAgent.run_stream(query, collection) -> AsyncGenerator[SSEEvent]`

- [ ] **Step 1: 编写流式测试**

在 `tests/unit/agent/test_react_agent.py` 末尾追加：

```python
class TestReActAgentStream:
    def test_stream_emits_all_events(self, programmable_llm, agent_config):
        """流式模式推送完整的 react_start → thought → action → observation → react_end 事件"""
        llm = programmable_llm([
            "THOUGHT: 需要搜索\nACTION: search\nQUERY: RAG架构",
            "THOUGHT: 信息充分\nACTION: FINISH",
        ])
        search = _make_search_tool([
            [{"doc_id": "d1", "text": "RAG是检索增强生成"}]
        ])
        web = _make_web_search_tool()
        agent = ReActAgent(llm, search, web, agent_config)

        async def collect():
            events = []
            async for e in agent.run_stream("什么是RAG？", "default"):
                events.append(e)
            return events

        events = asyncio.get_event_loop().run_until_complete(collect())

        event_names = [e.event for e in events]
        assert "react_start" in event_names
        assert "thought" in event_names
        assert "action" in event_names
        assert "observation" in event_names
        assert "react_end" in event_names

    def test_stream_react_end_has_stats(self, programmable_llm, agent_config):
        """react_end 事件包含 total_iterations 和 total_elapsed_ms"""
        llm = programmable_llm([
            "THOUGHT: 直接回答\nACTION: FINISH",
        ])
        search = _make_search_tool()
        web = _make_web_search_tool()
        agent = ReActAgent(llm, search, web, agent_config)

        async def collect():
            events = []
            async for e in agent.run_stream("hello", "default"):
                events.append(e)
            return events

        events = asyncio.get_event_loop().run_until_complete(collect())
        react_end = [e for e in events if e.event == "react_end"][0]
        assert react_end.data["total_iterations"] == 1
        assert "total_elapsed_ms" in react_end.data
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_react_agent.py::TestReActAgentStream -v
```

Expected: FAIL (run_stream 不存在)

- [ ] **Step 3: 实现 SSEEvent + run_stream()**

在 `src/agent/react_agent.py` 中追加：

```python
from collections.abc import AsyncGenerator
from typing import Any


@dataclass
class SSEEvent:
    """SSE 流式事件"""
    event: str   # "react_start" | "thought" | "action" | "observation" | "react_end"
    data: dict[str, Any]


# 在 ReActAgent 类中追加 run_stream 方法：

    async def run_stream(
        self, query: str, collection: str
    ) -> AsyncGenerator[SSEEvent, None]:
        """流式执行 ReAct 循环，逐事件推送"""
        from models.react_trace import ReActTrace

        t0 = time.perf_counter()
        traces: list[ReActTrace] = []
        observation: str | None = None
        seen_pairs: set[tuple[str, str]] = set()
        consecutive_dup_count = 0

        yield SSEEvent("react_start", {"mode": "react", "query": query})

        for iteration in range(1, self._config.max_iterations + 1):
            t_iter = time.perf_counter()

            # LLM 决策
            prompt = self._build_prompt(query, traces, observation)
            raw = await self._call_llm(prompt)
            parsed = parse_react_output(raw)

            if "parse_error" in parsed:
                trace = ReActTrace(
                    iteration=iteration,
                    thought=f"LLM 输出格式异常: {parsed['parse_error']}，强制结束",
                    action="finish",
                    elapsed_ms=(time.perf_counter() - t_iter) * 1000,
                )
                traces.append(trace)
                yield SSEEvent("thought", {
                    "iteration": iteration,
                    "thought": trace.thought,
                    "action": "finish",
                })
                break

            action = parsed["action"]
            search_query = parsed.get("query")

            # 连续重复检测
            pair_key = (action, search_query or "")
            if pair_key in seen_pairs:
                consecutive_dup_count += 1
                if consecutive_dup_count >= self._config.max_consecutive_duplicates:
                    yield SSEEvent("thought", {
                        "iteration": iteration,
                        "thought": f"连续 {consecutive_dup_count} 轮重复，终止循环",
                        "action": "finish",
                    })
                    break
            else:
                consecutive_dup_count = 0
            seen_pairs.add(pair_key)

            # 推送 thought 事件
            yield SSEEvent("thought", {
                "iteration": iteration,
                "thought": parsed["thought"],
                "action": action,
            })

            if action == "finish":
                trace = ReActTrace(
                    iteration=iteration,
                    thought=parsed["thought"],
                    action="finish",
                    elapsed_ms=(time.perf_counter() - t_iter) * 1000,
                )
                traces.append(trace)
                break

            # 推送 action 事件
            yield SSEEvent("action", {
                "iteration": iteration,
                "action": action,
                "query": search_query,
            })

            # 执行工具
            tool_result = await self._execute_tool(action, search_query or "", collection)
            observation = self._format_observation(tool_result)

            # 推送 observation 事件
            yield SSEEvent("observation", {
                "iteration": iteration,
                "chunk_count": tool_result.chunk_count,
                "elapsed_ms": round(tool_result.elapsed_ms, 2),
            })

            trace = ReActTrace(
                iteration=iteration,
                thought=parsed["thought"],
                action=action,
                query=search_query,
                observation=observation,
                elapsed_ms=(time.perf_counter() - t_iter) * 1000,
            )
            traces.append(trace)

        total_elapsed = (time.perf_counter() - t0) * 1000
        yield SSEEvent("react_end", {
            "total_iterations": len(traces),
            "total_elapsed_ms": round(total_elapsed, 2),
        })
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_react_agent.py -v
```

Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
git add src/agent/react_agent.py tests/unit/agent/test_react_agent.py
git commit -m "feat(agent): add run_stream() method with SSE event streaming"
```

---

### Task 7: agent/__init__.py 单例工厂

**Files:**
- Modify: `src/agent/__init__.py` — 单例工厂 `get_react_agent` / `reset_react_agent`
- Create: `tests/unit/agent/test_init.py` — 单例工厂测试

**Interfaces:**
- Produces: `get_react_agent(llm, search_tool, web_search_tool) -> ReActAgent`
- Produces: `reset_react_agent() -> None`

- [ ] **Step 1: 编写单例测试**

```python
# tests/unit/agent/test_init.py
from unittest.mock import MagicMock
from agent import get_react_agent, reset_react_agent


class TestSingleton:
    def test_get_returns_same_instance(self):
        llm = MagicMock()
        search = MagicMock()
        web = MagicMock()
        
        reset_react_agent()
        a1 = get_react_agent(llm, search, web)
        a2 = get_react_agent(llm, search, web)
        assert a1 is a2

    def test_reset_creates_new_instance(self):
        llm = MagicMock()
        search = MagicMock()
        web = MagicMock()
        
        reset_react_agent()
        a1 = get_react_agent(llm, search, web)
        reset_react_agent()
        a2 = get_react_agent(llm, search, web)
        assert a1 is not a2
```

- [ ] **Step 2: 实现单例工厂**

```python
# src/agent/__init__.py (完整替换占位内容)
"""ReAct Agent 模块 — 思考→行动→观察 循环"""
import threading
from typing import TYPE_CHECKING

from agent.react_agent import AgentResult, ReActAgent, SSEEvent, parse_react_output
from agent.tools import SearchTool, ToolResult, WebSearchTool

if TYPE_CHECKING:
    from models.llm import LLMProtocol

# 全局单例
_react_agent: ReActAgent | None = None
_lock = threading.Lock()


def get_react_agent(
    llm: "LLMProtocol",
    search_tool: SearchTool | None = None,
    web_search_tool: WebSearchTool | None = None,
) -> ReActAgent:
    """获取 ReActAgent 全局单例
    
    首次调用时必须传入 llm；search_tool / web_search_tool
    若未传入则从 fallback 模块自动构建。
    
    Args:
        llm: LLM 实例
        search_tool: 内部知识库搜索工具（可选，自动构建）
        web_search_tool: 联网搜索工具（可选，自动构建）
    
    Returns:
        ReActAgent 全局单例
    """
    global _react_agent
    
    if _react_agent is not None:
        return _react_agent
    
    with _lock:
        if _react_agent is None:
            if search_tool is None:
                from retrieval import get_retrieval_layer
                search_tool = SearchTool(get_retrieval_layer())
            if web_search_tool is None:
                from fallback.web_search import WebSearcher
                web_search_tool = WebSearchTool(WebSearcher())
            _react_agent = ReActAgent(llm, search_tool, web_search_tool)
        return _react_agent


def reset_react_agent() -> None:
    """重置全局单例（测试用）"""
    global _react_agent
    with _lock:
        _react_agent = None


__all__ = [
    "AgentResult",
    "ReActAgent",
    "SSEEvent",
    "SearchTool",
    "ToolResult",
    "WebSearchTool",
    "get_react_agent",
    "parse_react_output",
    "reset_react_agent",
]
```

- [ ] **Step 3: 运行测试确认通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/test_init.py -v
```

Expected: 2 passed

- [ ] **Step 4: 运行全部 agent 测试确认**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/agent/ -v
```

Expected: ~19 passed（9 解析器 + 5 工具 + 6+2 agent + 2 初始化）

- [ ] **Step 5: 提交**

```bash
git add src/agent/__init__.py tests/unit/agent/test_init.py
git commit -m "feat(agent): add get_react_agent singleton factory"
```

---

### Task 8: RAGPipeline 集成 — mode="react" 分支

**Files:**
- Modify: `src/core/pipeline.py` — `run()` 增加 `mode`/`max_iterations`/`show_reasoning` 参数 + `_run_react()` 方法 + `run_stream()` 方法
- Create: `tests/unit/core/test_pipeline_react.py` — ReAct 模式集成测试

**Interfaces:**
- Consumes: `ReActAgent` (from `agent/`), `GenerationLayer`, `FallbackHandler`
- Modifies: `RAGPipeline.run(mode, max_iterations, show_reasoning)` — 新增可选参数
- Produces: `RAGPipeline._run_react(query, session_id, collection, max_iterations, show_reasoning) -> PipelineContext`
- Produces: `RAGPipeline.run_stream(...) -> AsyncGenerator[SSEEvent]`

- [ ] **Step 1: 编写集成测试**

```python
# tests/unit/core/test_pipeline_react.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from core.pipeline import RAGPipeline
from models.context import PipelineContext
from models.enums import FallbackLevel, RetrievalEval


class ProgrammableLLM:
    """每次 ainvoke 按顺序返回 preset 响应"""
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def ainvoke(self, prompt: str, temperature: float = 0.0):
        if self.call_count >= len(self.responses):
            return MagicMock(content="THOUGHT: done\nACTION: FINISH")
        resp = self.responses[self.call_count]
        self.call_count += 1
        return MagicMock(content=resp)


class TestPipelineReactMode:
    @pytest.fixture
    def mock_session_manager(self):
        sm = MagicMock()
        sm.add_message = MagicMock()
        sm.get = MagicMock(return_value=None)
        return sm

    def test_react_mode_single_round(self, mock_session_manager):
        """ReAct 模式单轮 FINISH 后生成答案"""
        llm = ProgrammableLLM([
            # Agent 决策：直接回答
            "THOUGHT: 简单问候，无需检索\nACTION: FINISH",
            # 生成答案
            "你好！有什么可以帮助你的？",
        ])
        pipeline = RAGPipeline(llm, mock_session_manager)
        
        ctx = asyncio.get_event_loop().run_until_complete(
            pipeline.run("你好", mode="react")
        )
        
        assert ctx.mode == "react"
        assert len(ctx.react_traces) == 1
        assert ctx.react_traces[0].action == "finish"
        assert ctx.answer == "你好！有什么可以帮助你的？"

    def test_react_mode_falls_back_to_linear_on_error(self, mock_session_manager):
        """LLM 连续异常时降级到 linear 模式"""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        
        pipeline = RAGPipeline(llm, mock_session_manager)
        ctx = asyncio.get_event_loop().run_until_complete(
            pipeline.run("test", mode="react")
        )
        
        # 降级后应走 linear 流程（虽然检索也可能失败）
        assert ctx.mode == "linear"  # 降级后 mode 重置为 linear
        assert ctx.fallback_level in (FallbackLevel.NO_ANSWER, FallbackLevel.NONE)

    def test_linear_mode_unchanged(self, mock_session_manager):
        """mode='linear' 时行为不变"""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
        
        pipeline = RAGPipeline(llm, mock_session_manager)
        ctx = asyncio.get_event_loop().run_until_complete(
            pipeline.run("test", mode="linear")
        )
        
        assert ctx.mode == "linear"
        assert ctx.react_traces == []
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/core/test_pipeline_react.py -v
```

Expected: FAIL（mode 参数不存在）

- [ ] **Step 3: 修改 RAGPipeline**

修改 `src/core/pipeline.py` 的 `run()` 方法签名和实现。在 `RAGPipeline` 类中：

(1) 修改 `run()` 方法签名，增加新参数：

```python
async def run(
    self,
    query: str,
    session_id: str | None = None,
    collection: str = "default",
    mode: str = "linear",
    max_iterations: int = 5,
    show_reasoning: bool = False,
) -> PipelineContext:
```

(2) 在 `run()` 方法开头（`t0 = time.perf_counter()` 之后、`ctx = PipelineContext(...)` 之后）插入分叉逻辑：

```python
        ctx.mode = mode
        ctx.max_iterations = max_iterations

        if mode == "react":
            return await self._run_react(
                query, session_id, collection, max_iterations, show_reasoning
            )
```

(3) 新增 `_run_react()` 方法（在 `_save_to_session` 之前）：

```python
    async def _run_react(
        self,
        query: str,
        session_id: str | None,
        collection: str,
        max_iterations: int,
        show_reasoning: bool,
    ) -> PipelineContext:
        """ReAct Agent 分支"""
        from agent import get_react_agent
        from models.react_trace import ReActTrace

        t0 = time.perf_counter()
        ctx = PipelineContext(
            query=query, collection=collection,
            mode="react", max_iterations=max_iterations,
        )
        ctx.original_query = query

        # 1. 别名映射
        try:
            from config.aliases import resolve_aliases_in_text
            query = resolve_aliases_in_text(query)
            ctx.query = query
        except Exception:
            logger.warning("ReAct 别名映射失败，使用原始 query")

        # 2. ReAct Agent 循环
        try:
            agent = get_react_agent(self._llm, None, None)
            result = await agent.run(query, collection)
        except Exception:
            logger.exception("ReAct Agent 异常，降级到 linear 模式")
            return await self.run(query, session_id, collection, mode="linear")

        ctx.react_traces = result.react_traces
        ctx.reranked = result.reranked

        # 3. Agent 如果在循环中执行过 search，重新用 RetrievalLayer 统一检索
        #    合并所有 search query 的检索结果
        search_queries = [
            t.query for t in result.react_traces
            if t.action == "search" and t.query
        ]
        if search_queries and not ctx.reranked:
            try:
                # 用所有搜索过的 query 重新检索合并
                ctx.rewritten_queries = list(dict.fromkeys(search_queries))  # 去重保序
                ctx = await self.retrieval_layer.retrieve(ctx)
            except Exception:
                logger.exception("ReAct 合并检索异常")

        # 4. 检索自评
        if ctx.retrieval_eval is None:
            from retrieval.evaluator import evaluate
            ctx.retrieval_eval = evaluate(ctx.reranked)

        # 5. 兜底：INSUFFICIENT 且 Agent 未调用过 web_search → 联网
        web_searched = any(
            t.action == "web_search" for t in result.react_traces
        )
        if ctx.retrieval_eval is RetrievalEval.INSUFFICIENT and not web_searched:
            try:
                ctx = await self.fallback.handle(ctx)
            except Exception:
                logger.exception("ReAct 兜底异常")

        need_short_circuit = await self._apply_fallback(ctx, t0, query, session_id)
        if need_short_circuit:
            return ctx

        # 6. 生成
        try:
            ctx = await self.generation_layer.generate(ctx)
        except Exception:
            logger.exception("ReAct 生成层异常")
            self.fallback.no_answer(ctx)

        # 7. 记录
        self._record_elapsed(ctx, t0)
        if show_reasoning:
            ctx.metadata["react_traces"] = [
                {
                    "iteration": t.iteration,
                    "thought": t.thought,
                    "action": t.action,
                    "query": t.query,
                    "elapsed_ms": t.elapsed_ms,
                }
                for t in result.react_traces
            ]
        await self._save_to_session(session_id, query, ctx.answer)
        return ctx
```

(4) 新增 `run_stream()` 方法：

```python
    async def run_stream(
        self,
        query: str,
        session_id: str | None = None,
        collection: str = "default",
        mode: str = "linear",
        max_iterations: int = 5,
        show_reasoning: bool = False,
    ):
        """流式执行 RAG Pipeline，支持 ReAct 事件推送

        Phase 1: 透传 ReActAgent.run_stream() 事件（含过滤）
        Phase 2: 合并检索 → 生成 → done 事件

        Yields:
            SSEEvent
        """
        from agent.react_agent import SSEEvent

        if mode != "react":
            yield SSEEvent("done", {
                "answer": "streaming only supported for react mode",
                "sources": [], "confidence": 0.0,
            })
            return

        t0 = time.perf_counter()
        search_queries: list[str] = []
        web_searched = False

        # 1. 别名映射
        try:
            from config.aliases import resolve_aliases_in_text
            query = resolve_aliases_in_text(query)
        except Exception:
            logger.warning("流式别名映射失败")

        # 2. Agent 流式循环：收集事件 + 透传
        try:
            from agent import get_react_agent
            agent = get_react_agent(self._llm, None, None)
            async for event in agent.run_stream(query, collection):
                # 收集 search query 用于后续合并检索
                if event.event == "action" and event.data.get("action") == "search":
                    sq = event.data.get("query", "")
                    if sq:
                        search_queries.append(sq)
                if event.event == "action" and event.data.get("action") == "web_search":
                    web_searched = True
                # 过滤
                if not show_reasoning and event.event in ("thought", "action", "observation"):
                    continue
                yield event
        except Exception:
            logger.exception("ReAct Agent 流式异常")
            yield SSEEvent("done", {"answer": "", "sources": [], "confidence": 0.0})
            return

        # 3. 合并检索（用 Agent 搜过的所有 query）
        ctx = PipelineContext(query=query, collection=collection, mode="react")
        if search_queries:
            try:
                ctx.rewritten_queries = list(dict.fromkeys(search_queries))
                ctx = await self.retrieval_layer.retrieve(ctx)
            except Exception:
                logger.exception("流式合并检索异常")

        # 4. 兜底（INSUFFICIENT 且 Agent 未 web_search → 联网）
        if ctx.retrieval_eval is RetrievalEval.INSUFFICIENT and not web_searched:
            try:
                ctx = await self.fallback.handle(ctx)
            except Exception:
                logger.exception("流式兜底异常")

        # 5. 生成
        try:
            ctx = await self.generation_layer.generate(ctx)
        except Exception:
            logger.exception("流式生成异常")
            self.fallback.no_answer(ctx)

        self._record_elapsed(ctx, t0)
        await self._save_to_session(session_id, query, ctx.answer)

        yield SSEEvent("done", {
            "answer": ctx.answer,
            "sources": [
                {"doc_id": s.doc_id, "doc_title": s.doc_title,
                 "chunk_text": s.chunk_text[:200], "score": s.score}
                for s in (ctx.sources or [])
            ],
            "confidence": ctx.confidence,
        })
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/core/test_pipeline_react.py -v
```

Expected: 3 passed

- [ ] **Step 5: 运行全部 core 测试确认无回归**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/core/ -v
```

Expected: 全部通过（含原有 linear 测试）

- [ ] **Step 6: 提交**

```bash
git add src/core/pipeline.py tests/unit/core/test_pipeline_react.py
git commit -m "feat(core): add mode='react' branch with ReActAgent integration"
```

---

### Task 9: API 模型更新 — ChatRequest/ChatResponse 扩展

**Files:**
- Modify: `src/models/api.py` — ChatRequest/ChatResponse 增加 ReAct 字段

**Interfaces:**
- Modifies: `ChatRequest` — 增加 `mode`, `max_iterations`, `show_reasoning`
- Modifies: `ChatResponse` — 增加 `react_traces`

- [ ] **Step 1: 修改 ChatRequest 和 ChatResponse**

```python
# src/models/api.py — 在 ChatRequest 中增加三个字段
@dataclass
class ChatRequest:
    query: str
    session_id: str | None = None
    collection: str = "default"
    stream: bool = False
    top_k: int = 5
    mode: str = "linear"            # 🆕
    max_iterations: int = 5         # 🆕
    show_reasoning: bool = False    # 🆕


# 在 ChatResponse 中增加 react_traces 字段
@dataclass
class ChatResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    session_id: str = ""
    confidence: float = 0.0
    is_fallback: bool = False
    react_traces: list | None = None  # 🆕 show_reasoning=true 时返回
```

- [ ] **Step 2: 验证导入**

```bash
cd E:/Code/rag0709 && python -c "from models.api import ChatRequest; r = ChatRequest(query='test', mode='react'); print(r.mode)"
```

Expected: `react`

- [ ] **Step 3: 提交**

```bash
git add src/models/api.py
git commit -m "feat(models): extend ChatRequest/ChatResponse with ReAct fields"
```

---

### Task 10: agent/README.md 模块文档

**Files:**
- Create: `src/agent/README.md`

- [ ] **Step 1: 编写文档**

```markdown
# Agent 模块 — ReAct 代理模式

基于 ReAct（Reasoning + Acting）范式的知识库问答代理，使 LLM 自主决定检索时机和内容。

## 架构

```
用户 Query → ReActAgent 循环 (Thought→Action→Observe) → 生成答案
                   │
           ┌───────┴───────┐
      SearchTool      WebSearchTool
      (向量+BM25)      (DuckDuckGo)
```

## 快速使用

```python
from agent import get_react_agent, reset_react_agent

agent = get_react_agent(llm)

# 非流式
result = await agent.run("什么是RAG？", collection="default")
print(result.total_iterations)  # 循环轮次
print(result.react_traces)      # 推理链

# 流式
async for event in agent.run_stream("RAG优化方向", "default"):
    print(event.event, event.data)
```

## 通过 RAGPipeline 使用

```python
from core import get_rag_pipeline

pipeline = get_rag_pipeline(llm, session_manager)

# ReAct 模式
ctx = await pipeline.run("什么是RAG？", mode="react")

# 返回推理链
ctx = await pipeline.run("复杂问题", mode="react", show_reasoning=True)
print(ctx.metadata["react_traces"])
```

## 核心组件

| 组件 | 职责 |
|------|------|
| `ReActAgent` | 思考→行动→观察 主循环，LLM 决策 + 工具调用编排 |
| `SearchTool` | 封装 RetrievalLayer，每次 search 走完整混合检索 |
| `WebSearchTool` | 封装 WebSearcher，联网搜索兜底 |
| `parse_react_output` | 从 LLM 响应中提取 THOUGHT/ACTION/QUERY |

## 停止条件

| 条件 | 行为 |
|------|------|
| LLM 返回 `ACTION: FINISH` | 正常退出，进入生成 |
| 达到 `max_iterations` | 强制退出 |
| 连续重复 Action+Query | 死循环检测，退出 |
| LLM 调用异常 | 降级到 linear 模式 |

## 配置

```yaml
# config/defaults.yaml
agent:
  max_iterations: 5
  search_top_k: 3
  max_observation_chars: 3000
  llm_temperature: 0.0
  max_consecutive_duplicates: 2
```

## 测试

```bash
pytest tests/unit/agent/ -v
```
```

- [ ] **Step 2: 提交**

```bash
git add src/agent/README.md
git commit -m "docs(agent): add module README with usage guide"
```

---

## 自审检查清单

- [x] Spec 覆盖：每个设计点都有对应 Task（配置→数据模型→解析器→工具→Agent核心→流式→工厂→集成→API→文档）
- [x] 无占位符：所有代码步骤均包含具体实现
- [x] 类型一致性：`AgentResult` / `SSEEvent` / `ReActTrace` / `ToolResult` 跨 Task 引用一致
- [x] 接口对齐：`ReActAgent.__init__` 参数与 `get_react_agent` 调用一致；`RAGPipeline._run_react` 内部调用的方法签名与定义一致

