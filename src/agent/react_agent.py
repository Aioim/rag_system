"""ReActAgent — 思考→行动→观察 循环"""
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.settings import AgentConfig
    from agent.tools import SearchTool, WebSearchTool, ToolResult

_logger = logging.getLogger(__name__)


# ---- 输出解析器 ---------------------------------------------------------------

_VALID_ACTIONS = frozenset({"search", "web_search", "finish"})

_RE_THOUGHT = re.compile(r"THOUGHT:\s*(.+?)(?=\n\s*ACTION:|\Z)", re.DOTALL | re.IGNORECASE)
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


# ---- 数据模型 ---------------------------------------------------------------

@dataclass
class AgentResult:
    """ReAct Agent 执行结果"""
    reranked: list = field(default_factory=list)     # list[Chunk]
    react_traces: list = field(default_factory=list)  # list[ReActTrace]
    total_iterations: int = 0
    total_elapsed_ms: float = 0.0


@dataclass
class SSEEvent:
    """SSE 流式事件"""
    event: str   # "react_start" | "thought" | "action" | "observation" | "react_end"
    data: dict[str, Any]


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
        llm: Any,
        search_tool: Any,
        web_search_tool: Any,
        config: Any = None,
    ) -> None:
        from config import settings

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
        last_pair_key: tuple[str, str] | None = None
        consecutive_count = 0

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

            # 3. 连续重复检测（仅跟踪连续相同的 ACTION+QUERY）
            pair_key = (action, search_query or "")
            if pair_key == last_pair_key:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_pair_key = pair_key

            if consecutive_count > 1 and consecutive_count >= self._config.max_consecutive_duplicates:
                trace = ReActTrace(
                    iteration=iteration,
                    thought=f"连续 {consecutive_count} 轮重复 {action}:{search_query}，终止循环",
                    action="finish",
                    elapsed_ms=(time.perf_counter() - t_iter) * 1000,
                )
                traces.append(trace)
                break

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

            # 5. 最后一轮强制结束（不再执行工具）
            if iteration == self._config.max_iterations:
                trace = ReActTrace(
                    iteration=iteration,
                    thought=f"已达到最大迭代次数 ({self._config.max_iterations})，强制结束",
                    action="finish",
                    elapsed_ms=(time.perf_counter() - t_iter) * 1000,
                )
                traces.append(trace)
                break

            # 6. 执行工具 + 构建 Observation
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

    async def run_stream(
        self, query: str, collection: str
    ) -> AsyncGenerator[SSEEvent, None]:
        """流式执行 ReAct 循环，逐事件推送"""
        from models.react_trace import ReActTrace

        t0 = time.perf_counter()
        traces: list[ReActTrace] = []
        observation: str | None = None
        last_pair_key: tuple[str, str] | None = None
        consecutive_count = 0

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

            # 连续重复检测（仅跟踪连续相同的 ACTION+QUERY，与 run() 一致）
            pair_key = (action, search_query or "")
            if pair_key == last_pair_key:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_pair_key = pair_key

            if consecutive_count > 1 and consecutive_count >= self._config.max_consecutive_duplicates:
                yield SSEEvent("thought", {
                    "iteration": iteration,
                    "thought": f"连续 {consecutive_count} 轮重复 {action}:{search_query}，终止循环",
                    "action": "finish",
                })
                break

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

            # 最后一轮强制结束
            if iteration == self._config.max_iterations:
                trace = ReActTrace(
                    iteration=iteration,
                    thought=f"已达到最大迭代次数 ({self._config.max_iterations})，强制结束",
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

    # ---- 内部方法 ----------------------------------------------------------

    def _build_prompt(
        self, query: str, traces: list, observation: str | None
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
        except Exception as e:
            _logger.error("ReActAgent LLM 调用失败: %s", e)
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
            from agent.tools import ToolResult
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

    @staticmethod
    async def _collect_reranked(traces: list) -> list:
        """从 traces 中收集检索结果，由 RAGPipeline 统一处理。

        注意：实际 reranked chunks 由上层 RAGPipeline 通过
        search traces 的 query 重新检索获得（支持去重）。
        此处返回空列表，由 RAGPipeline 统一处理。
        """
        return []  # 由 RAGPipeline 的 _merge_react_results 统一处理
