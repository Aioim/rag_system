"""共享枚举类型"""
from enum import StrEnum


class Intent(StrEnum):
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    COMPARE = "compare"
    LOOKUP = "lookup"


class RetrievalEval(StrEnum):
    SUFFICIENT = "sufficient"
    NEED_MORE = "need_more"
    INSUFFICIENT = "insufficient"


class FallbackLevel(StrEnum):
    """兜底级别 — 从"未触发"到"诚实告知"依次升级"""
    NONE = "none"               # 未触发兜底
    PARTIAL = "partial"         # 资料不足但尝试生成（NEED_MORE）
    WEB_SEARCH = "web_search"   # 触发联网搜索
    NO_ANSWER = "no_answer"     # 诚实告知，无法回答


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    DONE = "done"
    FAILED = "failed"
