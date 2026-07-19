"""Document 测试"""
from pathlib import Path

from models.document import Document
from models.enums import DocumentStatus


class TestDocument:
    def test_minimal(self):
        doc = Document(doc_id="d1", source_path=Path("test.pdf"), file_type="pdf")
        assert doc.status == DocumentStatus.PENDING
        assert doc.raw_text == ""
        assert doc.created_at is not None

    def test_full(self):
        doc = Document(
            doc_id="d2", source_path=Path("r.docx"), file_type="docx",
            title="报告", raw_text="# 内容", collection="tech",
            status=DocumentStatus.DONE, metadata={"pages": 10},
        )
        assert doc.title == "报告"
        assert doc.collection == "tech"
        assert doc.status == DocumentStatus.DONE
        assert doc.metadata["pages"] == 10
