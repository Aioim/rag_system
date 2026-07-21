"""JSON 提取工具 — 已迁移至 src/utils/json_utils.py

保留此文件仅用于向后兼容，新代码请使用：
    from utils.json_utils import extract_json_container
"""

from utils.json_utils import (  # noqa: F401 — 向后兼容重导出
    extract_json_container,
    json_dumps_compact,
    safe_json_dumps,
    try_parse_json,
)
