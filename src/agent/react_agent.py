"""ReActAgent — 思考→行动→观察 循环"""
import re
from typing import Any


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
