"""ReAct Agent 模块 — 思考->行动->观察 循环"""
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

    agent = _react_agent
    if agent is not None:
        return agent

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
