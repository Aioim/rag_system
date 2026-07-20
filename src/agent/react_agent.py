"""ReActAgent — 思考→行动→观察 循环"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from models.react_trace import ReActTrace

if TYPE_CHECKING:
    from agent.tools import SearchTool, WebSearchTool
    from config.settings import AgentConfig
    from models.chunk import Chunk
    from models.llm import LLMProtocol

_logger = logging.getLogger(__name__)


# ---- 输出解析器 ---------------------------------------------------------------

_VALID_ACTIONS = frozenset({"search", "web_search", "finish"})

_RE_THOUGHT = re.compile(r"THOUGHT:\s*(.+?)(?=\n\s*ACTION:|\Z)", re.DOTALL | re.IGNORECASE)
_RE_ACTION = re.compile(r"ACTION:\s*(\w+)", re.IGNORECASE)
_RE_QUERY = re.compile(r"QUERY:\s*(.+)", re.IGNORECASE)


@dataclass
class ParseResult:
    """parse_react_output 解析成功"""
    thought: str
    action: str
    query: str | None


@dataclass
class ParseError:
    """parse_react_output 解析失败"""
    error: str


def parse_react_output(text: str) -> ParseResult | ParseError:
    """从 LLM 响应中提取 THOUGHT / ACTION / QUERY 三元组。

    Returns:
        ParseResult(thought, action, query) — 解析成功
        ParseError(error) — 解析失败
    """
    if not text or not text.strip():
        return ParseError("empty input")

    # 提取 THOUGHT
    m_thought = _RE_THOUGHT.search(text)
    if not m_thought:
        return ParseError("missing THOUGHT field")
    thought = " ".join(m_thought.group(1).strip().split())

    # 提取 ACTION
    m_action = _RE_ACTION.search(text)
    if not m_action:
        return ParseError("missing ACTION field")
    action = m_action.group(1).strip().lower()
    if action not in _VALID_ACTIONS:
        return ParseError(f"unknown action: {action!r}")

    # FINISH 不需要 QUERY
    if action == "finish":
        return ParseResult(thought=thought, action="finish", query=None)

    # search / web_search 必须有 QUERY
    m_query = _RE_QUERY.search(text)
    if not m_query:
        return ParseError(f"missing QUERY field for action={action!r}")
    query = " ".join(m_query.group(1).strip().split())

    return ParseResult(thought=thought, action=action, query=query)


# ---- 数据模型 ---------------------------------------------------------------

@dataclass
class AgentResult:
    """ReAct Agent 执行结果"""
    reranked: list[Chunk] = field(default_factory=list)  # 实际由 RAGPipeline 填充 Chunk 列表
    react_traces: list[ReActTrace] = field(default_factory=list)
    total_iterations: int = 0
    total_elapsed_ms: float = 0.0


@dataclass
class SSEEvent:
    """SSE 流式事件"""
    event: str   # "react_start" | "thought" | "action" | "observation" | "react_end"
    data: dict[str, Any]


@dataclass
class _LoopStep:
    """内部：ReAct 循环单步执行结果，run() 和 run_stream() 共享。

    除 trace 和 should_break 外，其余字段为循环状态传递（非 trace 已有字段）。
    thought/action/query/observation 通过 step.trace.* 访问，不在此重复存储。
    """
    trace: ReActTrace
    should_break: bool
    tool_result: Any = None                     # ToolResult（仅 tool_executed 时有值）
    next_pair_key: tuple[str, str] | None = None
    next_consecutive_count: int = 0


# ---- System Prompt ----------------------------------------------------------

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


# ---- ReActAgent ------------------------------------------------------------

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
        llm: LLMProtocol,
        search_tool: SearchTool,
        web_search_tool: WebSearchTool,
        config: AgentConfig | None = None,
    ) -> None:
        from config import settings

        self._llm = llm
        self._search = search_tool
        self._web = web_search_tool
        self._config: AgentConfig = config or settings.agent

    # ---- 主入口 ------------------------------------------------------------

    async def run(self, query: str, collection: str) -> AgentResult:
        """执行 ReAct 循环，返回 AgentResult"""
        t0 = time.perf_counter()
        traces: list[ReActTrace] = []
        observation: str | None = None
        pair_key: tuple[str, str] | None = None
        consecutive_count = 0

        for iteration in range(1, self._config.max_iterations + 1):
            step = await self._execute_step(
                query=query,
                collection=collection,
                traces=traces,
                observation=observation,
                iteration=iteration,
                last_pair_key=pair_key,
                consecutive_count=consecutive_count,
            )
            traces.append(step.trace)
            if step.should_break:
                break
            observation = step.trace.observation
            pair_key = step.next_pair_key
            consecutive_count = step.next_consecutive_count

        return AgentResult(
            reranked=[],  # 由 RAGPipeline._run_react 通过 search_queries 重新检索填充
            react_traces=traces,
            total_iterations=len(traces),
            total_elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    async def run_stream(
        self, query: str, collection: str
    ) -> AsyncGenerator[SSEEvent, None]:
        """流式执行 ReAct 循环，逐事件推送"""
        t0 = time.perf_counter()
        traces: list[ReActTrace] = []
        observation: str | None = None
        pair_key: tuple[str, str] | None = None
        consecutive_count = 0

        yield SSEEvent("react_start", {"mode": "react", "query": query})

        for iteration in range(1, self._config.max_iterations + 1):
            step = await self._execute_step(
                query=query,
                collection=collection,
                traces=traces,
                observation=observation,
                iteration=iteration,
                last_pair_key=pair_key,
                consecutive_count=consecutive_count,
            )
            traces.append(step.trace)

            # 推送 thought 事件（所有分支都推送，包括 parse_error/duplicate/max_iterations/finish）
            yield SSEEvent("thought", {
                "iteration": iteration,
                "thought": step.trace.thought,
                "action": step.trace.action,
            })

            if step.should_break:
                break

            # 推送 action + observation 事件（仅非终止步骤）
            yield SSEEvent("action", {
                "iteration": iteration,
                "action": step.trace.action,
                "query": step.trace.query,
            })

            tr = step.tool_result
            if tr is not None:
                yield SSEEvent("observation", {
                    "iteration": iteration,
                    "chunk_count": tr.chunk_count,
                    "elapsed_ms": round(tr.elapsed_ms, 2),
                })

            observation = step.trace.observation
            pair_key = step.next_pair_key
            consecutive_count = step.next_consecutive_count

        total_elapsed = (time.perf_counter() - t0) * 1000
        yield SSEEvent("react_end", {
            "total_iterations": len(traces),
            "total_elapsed_ms": round(total_elapsed, 2),
        })

    # ---- 核心步骤 ----------------------------------------------------------

    async def _execute_step(
        self,
        query: str,
        collection: str,
        traces: list[ReActTrace],
        observation: str | None,
        iteration: int,
        last_pair_key: tuple[str, str] | None,
        consecutive_count: int,
    ) -> _LoopStep:
        """执行 ReAct 循环的单个步骤：LLM 决策 → 解析 → 验证 → 工具执行。

        run() 和 run_stream() 共享此方法，确保核心逻辑一致。
        """
        t_iter = time.perf_counter()

        # 1. LLM 决策 + 解析
        prompt = self._build_prompt(query, traces, observation)
        raw = await self._call_llm(prompt)
        parsed = parse_react_output(raw)

        # 2. 计算重复检测 key
        if isinstance(parsed, ParseResult):
            pair_key = (parsed.action, parsed.query or "")
        else:
            pair_key = ("__error__", parsed.error)
        if pair_key == last_pair_key:
            consecutive_count += 1
        else:
            consecutive_count = 1

        # 3. 终止条件检查
        early = self._check_step_termination(parsed, iteration, t_iter, pair_key, consecutive_count)
        if early is not None:
            return early

        # 4. 工具执行（parsed 已确认为 ParseResult）
        return await self._build_tool_step(parsed, collection, iteration, t_iter, pair_key, consecutive_count)

    def _check_step_termination(
        self,
        parsed: ParseResult | ParseError,
        iteration: int,
        t_iter: float,
        pair_key: tuple[str, str],
        consecutive_count: int,
    ) -> _LoopStep | None:
        """检查是否需要提前终止循环（解析失败/重复/FINISH/达到最大轮数）。

        Returns:
            _LoopStep 如果需要终止，None 表示继续执行工具。
        """
        # 解析失败 → 强制 FINISH
        if isinstance(parsed, ParseError):
            trace = ReActTrace(
                iteration=iteration,
                thought=f"LLM 输出格式异常: {parsed.error}，强制结束",
                action="finish",
                elapsed_ms=(time.perf_counter() - t_iter) * 1000,
            )
            return _LoopStep(
                trace=trace, should_break=True,
                next_pair_key=pair_key,
                next_consecutive_count=consecutive_count,
            )

        # 连续重复检测
        if consecutive_count > 1 and consecutive_count >= self._config.max_consecutive_duplicates:
            trace = ReActTrace(
                iteration=iteration,
                thought=f"连续 {consecutive_count} 轮重复 {parsed.action}:{parsed.query}，终止循环",
                action="finish",
                elapsed_ms=(time.perf_counter() - t_iter) * 1000,
            )
            return _LoopStep(
                trace=trace, should_break=True,
                next_pair_key=pair_key,
                next_consecutive_count=consecutive_count,
            )

        # FINISH → 退出
        if parsed.action == "finish":
            trace = ReActTrace(
                iteration=iteration,
                thought=parsed.thought,
                action="finish",
                elapsed_ms=(time.perf_counter() - t_iter) * 1000,
            )
            return _LoopStep(
                trace=trace, should_break=True,
                next_pair_key=pair_key,
                next_consecutive_count=consecutive_count,
            )

        # 最后一轮强制结束
        if iteration == self._config.max_iterations:
            trace = ReActTrace(
                iteration=iteration,
                thought=f"已达到最大迭代次数 ({self._config.max_iterations})，强制结束",
                action="finish",
                elapsed_ms=(time.perf_counter() - t_iter) * 1000,
            )
            return _LoopStep(
                trace=trace, should_break=True,
                next_pair_key=pair_key,
                next_consecutive_count=consecutive_count,
            )

        return None

    async def _build_tool_step(
        self,
        parsed: ParseResult,
        collection: str,
        iteration: int,
        t_iter: float,
        pair_key: tuple[str, str],
        consecutive_count: int,
    ) -> _LoopStep:
        """执行工具并构建 Observation 步骤"""
        tool_result = await self._execute_tool(parsed.action, parsed.query or "", collection)
        observation_str = self._format_observation(tool_result)

        trace = ReActTrace(
            iteration=iteration,
            thought=parsed.thought,
            action=parsed.action,
            query=parsed.query,
            observation=observation_str,
            elapsed_ms=(time.perf_counter() - t_iter) * 1000,
        )

        return _LoopStep(
            trace=trace, should_break=False,
            tool_result=tool_result,
            next_pair_key=pair_key,
            next_consecutive_count=consecutive_count,
        )

    # ---- 内部方法 ----------------------------------------------------------

    def _build_prompt(
        self, query: str, traces: list[ReActTrace], observation: str | None
    ) -> str:
        """构建本轮 LLM 调用的完整 prompt"""
        parts = [_SYSTEM_PROMPT, "", f"用户问题: {query}"]

        # 注入历史痕迹（含每轮的 observation）
        for t in traces:
            parts.append("")
            parts.append(f"THOUGHT: {t.thought}")
            parts.append(f"ACTION: {t.action}")
            if t.query:
                parts.append(f"QUERY: {t.query}")
            if t.observation:
                parts.append(t.observation)

        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM，异常时返回空字符串"""
        try:
            result = await self._llm.ainvoke(
                prompt, temperature=self._config.llm_temperature
            )
            return result.content
        except Exception:
            _logger.exception("ReActAgent LLM 调用失败")
            return ""

    async def _execute_tool(
        self, action: str, query: str, collection: str
    ) -> Any:
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
            _logger.warning("工具执行失败 %s: %s", action, e)
            return ToolResult(
                tool=action, query=query, content=f"工具执行错误: {e}",
                chunk_count=0, elapsed_ms=0,
            )

    @staticmethod
    def _format_observation(result: Any) -> str:
        """格式化 Observation 文本"""
        return (
            f"OBSERVATION ({result.tool}, {result.chunk_count} chunks, "
            f"{result.elapsed_ms:.0f}ms):\n{result.content}"
        )

