# RAG 安全日志系统

为 RAG 知识库问答系统提供的安全日志模块，具备敏感数据脱敏、安全审计、性能监控等能力。

## 文件结构

```
logger/
├── __init__.py      # 模块导出 + 预配置日志器
├── lazy.py          # LazyLogger — 线程安全延迟初始化
├── core.py          # 装饰器 + RequestLogger + 安全事件
├── handlers.py      # HandlerFactory — 文件轮转处理器
├── formatters.py    # SecurityFormatter — 统一格式 + CRLF 清理
├── filters.py       # SensitiveDataFilter + SecurityAuditFilter
├── masking.py       # MaskingEngine — 正则脱敏引擎
├── helpers.py       # 调用栈追踪工具
├── metrics.py       # LogMetrics — 运行时计数器
└── README.md
```

## 快速开始

```python
from logger import logger, security_logger

logger.info("RAG 服务启动")
logger.warning("检索超时")
security_logger.info("用户认证成功")
```

## 预配置日志器

| 实例 | 输出 | 日志文件 |
|------|------|---------|
| `logger` | 控制台 + `rag_service.log` + `error.log` | 主日志 |
| `security_logger` | 控制台 + `security.log` | 安全审计 |

## 高级功能

### 性能监控

```python
from logger import log_performance

@log_performance(threshold_ms=100)
def retrieve(query: str):
    ...
```

### 步骤跟踪

```python
from logger import log_step

@log_step("文档分块")
def chunk_document(doc):
    ...
```

### 执行时长

```python
from logger import log_duration

with log_duration("Embedding 生成"):
    embeddings = model.encode(texts)
```

### 安全事件

```python
from logger import log_security_event

log_security_event(
    action="document_access",
    user="user_001",
    resource="doc_123",
    status="success",
    details={"collection": "tech_docs"}
)
```

### 异常记录

```python
from logger import log_exception

try:
    results = milvus.search(query)
except Exception:
    log_exception(context="Milvus 检索")
```

### 敏感数据脱敏

```python
from logger import mask_sensitive_data

# 自动脱敏（logger 内部已集成）
logger.info("password=secret123")  # → password=******

# 手动脱敏
clean = mask_sensitive_data("token=abc.xyz")
```

脱敏覆盖：密码、token、API key、手机号、身份证号、银行卡号、信用卡、邮箱。

## 自定义日志器

```python
from logger import LazyLogger

retrieval_log = LazyLogger.get(
    "retrieval",
    separate_log_file="retrieval.log"
)
```

## 日志格式

```
2026-07-11 15:28:09 INFO     [retrieval.py:search:42] 检索完成 (45.2ms)
```

- 时间戳 | 级别 | [文件名:函数:行号] | 消息

## 配置

日志行为由 `config/settings.py` 中的 `LogConfig` 控制：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `log_level` | `INFO` | 日志级别 |
| `log_dir` | `PROJECT_ROOT/logs` | 日志目录 |
| `log_file` | `rag_service.log` | 主日志文件 |
| `max_bytes` | 10MB | 单文件最大尺寸 |
| `backup_count` | 7 | 轮转保留份数 |
| `quiet` | `False` | 静默模式（抑制启动横幅） |
| `enable_colors` | `False` | 控制台彩色输出 |
