"""Ingestion Pipeline 数据模型 — 从 models 导入共享类型，定义 ingestion 专有类型"""

from dataclasses import dataclass, field

from models.document import Document  # noqa: F401 — 重导出
from models.chunk import Chunk  # noqa: F401 — 重导出


@dataclass
class StageError:
    """Stage 执行错误"""
    stage: str
    error: str
    fatal: bool = False


@dataclass
class PipelineContext:
    """Ingestion Pipeline 贯穿全链路的数据容器

    注意：与 models.PipelineContext（在线 QA 版）不同，
    此版本用于离线文档处理链路（Document → Chunks → FAISS）。
    """
    document: Document
    chunks: list[Chunk] = field(default_factory=list)
    current_stage: str = ""
    status: str = "pending"             # pending → running → done / failed
    errors: list[StageError] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
