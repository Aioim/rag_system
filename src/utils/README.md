# Utils 模块 — 通用工具类

项目级工具函数集合，放置不隶属于任何业务模块的通用辅助代码。

## 文件结构

```
utils/
├── __init__.py      # 统一导出
└── json_utils.py    # JSON 解析与序列化工具
```

## JSON 工具 (`json_utils`)

```python
from utils.json_utils import (
    extract_json_container,  # 从 LLM 响应中提取 JSON 对象/数组
    try_parse_json,          # 安全解析 JSON，失败返回 None
    safe_json_dumps,         # 安全序列化，不可序列化对象降级为 str()
    json_dumps_compact,      # 紧凑序列化（无空格）
)
```

### extract_json_container

从 LLM 响应中提取最外层 JSON 容器（处理 Markdown 包裹、额外文本等情况）：

```python
# 快速路径：整个字符串就是合法 JSON
extract_json_container('[{"a": 1}]', "[", "]")  # → '[{"a": 1}]'

# 慢速路径：带 Markdown 代码块
extract_json_container('```json\n{"key": "val"}\n```')  # → '{"key": "val"}'

# 失败返回 None
extract_json_container("no json here")  # → None
```

### try_parse_json

自动尝试多种方式解析 JSON：

```python
try_parse_json('{"a": 1}')                 # → {"a": 1}
try_parse_json('prefix [1, 2, 3] suffix')  # → [1, 2, 3]
try_parse_json("not json at all")           # → None
```

### safe_json_dumps

对不可序列化对象自动降级为 `str()`：

```python
from datetime import datetime

safe_json_dumps({"time": datetime.now()})  # → '{"time": "2026-07-21 10:00:00"}'
```

### json_dumps_compact

无空格紧凑序列化：

```python
json_dumps_compact({"a": 1, "b": [2, 3]})  # → '{"a":1,"b":[2,3]}'
```

## 向后兼容

`from models.json_utils import extract_json_container` 仍然可用，但新代码请使用 `from utils.json_utils import ...`。
