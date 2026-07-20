"""JSON 提取工具 — 从 LLM 响应中提取 JSON 对象/数组"""

import json


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
