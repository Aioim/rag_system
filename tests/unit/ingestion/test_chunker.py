"""ChunkerStage + 三种 splitter 测试"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ingestion.context import Chunk, Document, PipelineContext
from ingestion.chunker import (
    ChunkerStage,
    FixedChunker,
    HierarchicalChunker,
    SemanticChunker,
)


def _make_mock_embedding_model():
    """创建一个假的 embedding model，返回随机向量"""
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts: [np.random.rand(1024).astype(np.float32)
                                    for _ in texts]
    )
    return model


# ---- FixedChunker ----

class TestFixedChunker:
    def test_basic_split(self):
        splitter = FixedChunker(chunk_size=100, overlap=20)
        text = "这是测试内容。" * 30  # ~180 字，大于 chunk_size
        chunks = splitter.splitter(text)

        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id
            assert c.doc_id == ""  # doc_id 由 ChunkerStage 填入
            assert len(c.text) > 0

    def test_short_text_single_chunk(self):
        splitter = FixedChunker(chunk_size=500, overlap=50)
        chunks = splitter.splitter("短文本")
        assert len(chunks) == 1

    def test_linked_list_links(self):
        splitter = FixedChunker(chunk_size=80, overlap=20)
        text = "测试内容。" * 50
        chunks = splitter.splitter(text)

        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            if i > 0:
                assert c.prev_chunk_id == chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                assert c.next_chunk_id == chunks[i + 1].chunk_id


# ---- SemanticChunker ----

class TestSemanticChunker:
    def test_basic_split(self):
        model = _make_mock_embedding_model()
        splitter = SemanticChunker(
            embedding_model=model,
            chunk_size=200,
            overlap=30,
            threshold_percentile=0.9,
            buffer_size=1,
        )
        text = (
            "Python 是一种解释型语言。" * 10
            + "\n\n"
            + "机器学习是人工智能的一个分支。" * 10
        )
        chunks = splitter.splitter(text)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_short_text(self):
        model = _make_mock_embedding_model()
        splitter = SemanticChunker(
            embedding_model=model,
            chunk_size=500,
            overlap=30,
        )
        chunks = splitter.splitter("很短的文本")
        assert len(chunks) == 1


# ---- HierarchicalChunker ----

class TestHierarchicalChunker:
    def test_split_by_headings(self):
        splitter = HierarchicalChunker(chunk_size=200, overlap=30)
        text = (
            "# 第一章\n\n这是第一章的内容。" * 5
            + "\n\n## 1.1 小节\n\n这是小节内容。" * 5
            + "\n\n# 第二章\n\n这是第二章的内容。" * 5
        )
        chunks = splitter.splitter(text)

        assert len(chunks) >= 2
        headings_found = any(
            "第一章" in c.metadata.get("heading_path", "") for c in chunks
        )
        assert headings_found


# ---- ChunkerStage ----

class TestChunkerStage:
    @pytest.mark.asyncio
    async def test_selects_fixed_strategy(self):
        """ChunkerStage 根据 settings 选择 splitter"""
        from config import settings

        original = settings.chunking.strategy
        settings.chunking.strategy = "fixed"

        try:
            stage = ChunkerStage()
            doc = Document(
                doc_id="d-1",
                source_path=Path("test.md"),
                file_type="md",
                raw_text="测试内容。" * 30,
            )
            ctx = PipelineContext(document=doc)
            result = await stage.run(ctx)

            assert len(result.chunks) > 0
            assert all(c.doc_id == "d-1" for c in result.chunks)
        finally:
            settings.chunking.strategy = original

    @pytest.mark.asyncio
    async def test_chunks_carry_doc_title_metadata(self):
        """chunk.metadata 写入 doc_title，供生成层引用来源展示"""
        from config import settings

        original = settings.chunking.strategy
        settings.chunking.strategy = "fixed"

        try:
            stage = ChunkerStage()
            doc = Document(
                doc_id="d-1",
                source_path=Path("员工手册.md"),
                file_type="md",
                title="员工手册",
                raw_text="测试内容。" * 30,
            )
            ctx = PipelineContext(document=doc)
            result = await stage.run(ctx)

            assert len(result.chunks) > 0
            assert all(
                c.metadata["doc_title"] == "员工手册" for c in result.chunks
            )
        finally:
            settings.chunking.strategy = original

    @pytest.mark.asyncio
    async def test_empty_text_records_error(self):
        stage = ChunkerStage()
        doc = Document(
            doc_id="d-1",
            source_path=Path("test.md"),
            file_type="md",
            raw_text="",
        )
        ctx = PipelineContext(document=doc)
        result = await stage.run(ctx)

        assert result.chunks == []
        assert len(result.errors) >= 1
        assert any("empty" in e.error.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_name_and_fatal(self):
        stage = ChunkerStage()
        assert stage.name == "chunker"
        assert stage.fatal is False
