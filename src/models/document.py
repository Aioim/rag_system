"""Document 数据模型"""
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from models.enums import DocumentStatus


@dataclass
class Document:
    doc_id: str
    source_path: Path
    file_type: str
    title: str = ""
    raw_text: str = ""
    collection: str = "default"
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
