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
        chunks = splitter.split(text)

        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id
            assert c.doc_id == ""  # doc_id 由 ChunkerStage 填入
            assert len(c.text) > 0

    def test_short_text_single_chunk(self):
        splitter = FixedChunker(chunk_size=500, overlap=50)
        chunks = splitter.split("短文本")
        assert len(chunks) == 1

    def test_linked_list_links(self):
        splitter = FixedChunker(chunk_size=80, overlap=20)
        text = "测试内容。" * 50
        chunks = splitter.split(text)

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
        chunks = splitter.split(text)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_short_text(self):
        model = _make_mock_embedding_model()
        splitter = SemanticChunker(
            embedding_model=model,
            chunk_size=500,
            overlap=30,
        )
        chunks = splitter.split("很短的文本")
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
        chunks = splitter.split(text)

        assert len(chunks) >= 2
        headings_found = any(
            "第一章" in c.metadata.get("heading_path", "") for c in chunks
        )
        assert headings_found

    def test_long_section_window_overlap_not_duplicated(self):
        """长 section 滑窗切分已含 overlap，后处理不应再拼前块尾部（复现文本重复 bug）"""
        splitter = HierarchicalChunker(chunk_size=10, overlap=4)
        # 36 个互不重复的字符，无标题 → 整体为单个长 section
        text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        chunks = splitter.split(text)

        assert chunks[0].text == "0123456789"
        # 滑窗第二窗 [6:16] 与第一窗已重叠 "6789"，不应再拼成 "67896789ABCDEF"
        assert chunks[1].text == "6789ABCDEF"
        # 所有 chunk 都应是原文的连续子串（任何重复拼接都会破坏该性质）
        for c in chunks:
            assert c.text in text

    def test_short_sections_still_get_cross_section_overlap(self):
        """短章节之间仍应拼接前块尾部 overlap（保留原有设计行为）"""
        splitter = HierarchicalChunker(chunk_size=100, overlap=4)
        text = "# A\n甲甲甲\n# B\n乙乙乙"

        chunks = splitter.split(text)

        assert len(chunks) == 2
        assert chunks[1].text.startswith(chunks[0].text[-4:])


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

    @pytest.mark.asyncio
    async def test_semantic_encode_runs_off_event_loop_thread(self):
        """SemanticChunker 的同步 encode 不应在事件循环线程执行（复现阻塞事件循环 bug）"""
        import threading

        from config import settings

        loop_thread = threading.current_thread()
        encode_threads: list[threading.Thread] = []

        def _record_thread(texts):
            encode_threads.append(threading.current_thread())
            return [np.random.rand(8).astype(np.float32) for _ in texts]

        model = MagicMock()
        model.encode = MagicMock(side_effect=_record_thread)

        original = settings.chunking.strategy
        settings.chunking.strategy = "semantic"
        try:
            stage = ChunkerStage(embedding_model=model)
            doc = Document(
                doc_id="d-1",
                source_path=Path("test.md"),
                file_type="md",
                raw_text="第一句。第二句。第三句。第四句。",
            )
            ctx = PipelineContext(document=doc)
            result = await stage.run(ctx)
        finally:
            settings.chunking.strategy = original

        assert len(result.chunks) > 0
        assert encode_threads, "encode 未被调用"
        assert all(t is not loop_thread for t in encode_threads), (
            "encode 在事件循环线程上执行，会阻塞事件循环"
        )
