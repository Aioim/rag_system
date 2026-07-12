"""PipelineContext、Document、Chunk、StageError 数据模型测试"""

from pathlib import Path

from ingestion.context import Chunk, Document, PipelineContext, StageError


class TestDocument:
    def test_minimal_construction(self):
        doc = Document(
            doc_id="doc-001",
            source_path=Path("/tmp/test.pdf"),
            file_type="pdf",
        )
        assert doc.doc_id == "doc-001"
        assert doc.file_type == "pdf"
        assert doc.title == ""
        assert doc.raw_text == ""
        assert doc.collection == "default"
        assert doc.metadata == {}

    def test_full_construction(self):
        doc = Document(
            doc_id="doc-002",
            source_path=Path("/tmp/report.docx"),
            file_type="docx",
            title="年度报告",
            raw_text="# 第一章\n内容...",
            collection="tech",
            metadata={"author": "张三", "pages": 10},
        )
        assert doc.title == "年度报告"
        assert doc.collection == "tech"
        assert doc.metadata["author"] == "张三"


class TestChunk:
    def test_minimal_construction(self):
        chunk = Chunk(
            chunk_id="c-001",
            doc_id="doc-001",
            text="这是一段测试文本",
            chunk_index=0,
        )
        assert chunk.chunk_id == "c-001"
        assert chunk.text == "这是一段测试文本"
        assert chunk.prev_chunk_id is None
        assert chunk.next_chunk_id is None
        assert chunk.context_summary is None
        assert chunk.embedding is None
        assert chunk.metadata == {}

    def test_linked_list(self):
        chunk = Chunk(
            chunk_id="c-mid",
            doc_id="d-1",
            text="中间段",
            chunk_index=2,
            prev_chunk_id="c-001",
            next_chunk_id="c-002",
        )
        assert chunk.prev_chunk_id == "c-001"
        assert chunk.next_chunk_id == "c-002"


class TestPipelineContext:
    def test_default_construction(self):
        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        ctx = PipelineContext(document=doc)
        assert ctx.document == doc
        assert ctx.chunks == []
        assert ctx.current_stage == ""
        assert ctx.status == "pending"
        assert ctx.errors == []
        assert ctx.metadata == {}

    def test_with_chunks(self):
        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        chunks = [
            Chunk(chunk_id="c-1", doc_id="d-1", text="A", chunk_index=0),
            Chunk(chunk_id="c-2", doc_id="d-1", text="B", chunk_index=1),
        ]
        ctx = PipelineContext(document=doc, chunks=chunks, status="running")
        assert len(ctx.chunks) == 2
        assert ctx.status == "running"


class TestStageError:
    def test_default_not_fatal(self):
        err = StageError(stage="parser", error="文件无法打开")
        assert err.stage == "parser"
        assert err.error == "文件无法打开"
        assert err.fatal is False

    def test_fatal_error(self):
        err = StageError(stage="parser", error="磁盘满", fatal=True)
        assert err.fatal is True
