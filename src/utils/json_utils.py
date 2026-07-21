"""JSON 工具集 — 解析、提取、安全序列化"""

import json
from typing import Any


def extract_json_container(
    raw: str, open_char: str = "{", close_char: str = "}"
) -> str | None:
    """从 LLM 响应中提取最外层 JSON 容器（对象或数组）。

    处理 LLM 输出中常见的 Markdown 包裹、额外文本等情况：
    - 优先尝试直接解析整个响应（快速路径）
    - 失败后降级到括号计数提取（处理前后缀文本）

    Args:
        raw: LLM 原始响应文本
        open_char: 起始括号（'{' 或 '['）
        close_char: 结束括号（'}' 或 ']'）

    Returns:
        提取到的 JSON 字符串，失败返回 None
    """
    stripped = raw.strip()
    # 快速路径：整个响应就是合法 JSON
    if stripped.startswith(open_char):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    # 慢速路径：括号计数提取最外层容器
    start = raw.find(open_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def try_parse_json(raw: str) -> Any | None:
    """尝试解析 JSON 字符串，失败返回 None（不抛异常）。

    自动尝试提取 JSON 对象和数组容器后再解析。

    Args:
        raw: 原始文本（可含 Markdown 包裹或前后缀）

    Returns:
        解析后的 Python 对象，失败返回 None
    """
    # 先尝试直接解析
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 对象
    for open_c, close_c in [("{", "}"), ("[", "]")]:
        container = extract_json_container(raw, open_c, close_c)
        if container:
            try:
                return json.loads(container)
            except json.JSONDecodeError:
                continue
    return None


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """安全序列化为 JSON 字符串，对不可序列化对象降级为 str()。

    Args:
        obj: 待序列化对象
        **kwargs: 传递给 json.dumps 的额外参数

    Returns:
        JSON 字符串
    """
    defaults = {"ensure_ascii": False, "default": str}
    defaults.update(kwargs)
    return json.dumps(obj, **defaults)


def json_dumps_compact(obj: Any) -> str:
    """紧凑 JSON 序列化（无空格、无 ASCII 转义）。

    Args:
        obj: 待序列化对象

    Returns:
        紧凑 JSON 字符串
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
