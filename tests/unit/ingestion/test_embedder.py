"""EmbedderStage 测试"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ingestion.context import Chunk, Document, PipelineContext
from ingestion.embedder import EmbedderStage


def _make_mock_embedding_model():
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts: [np.random.rand(1024).astype(np.float32)
                                    for _ in texts]
    )
    return model


class TestEmbedderStage:
    @pytest.mark.asyncio
    async def test_embeds_all_chunks(self):
        model = _make_mock_embedding_model()
        stage = EmbedderStage(embedding_model=model)

        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        chunks = [
            Chunk(chunk_id="c-1", doc_id="d-1", text="文本 A", chunk_index=0),
            Chunk(chunk_id="c-2", doc_id="d-1", text="文本 B", chunk_index=1),
            Chunk(chunk_id="c-3", doc_id="d-1", text="文本 C", chunk_index=2),
        ]
        ctx = PipelineContext(document=doc, chunks=chunks)

        result = await stage.run(ctx)

        for c in result.chunks:
            assert c.embedding is not None
            assert len(c.embedding) == 1024

    @pytest.mark.asyncio
    async def test_skips_already_embedded_chunks(self):
        """已有 embedding 的 chunk 应跳过（幂等）"""
        model = _make_mock_embedding_model()
        stage = EmbedderStage(embedding_model=model)

        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        pre_embedded = Chunk(
            chunk_id="c-done", doc_id="d-1", text="已处理",
            chunk_index=0, embedding=[0.5] * 1024,
        )
        new_chunk = Chunk(
            chunk_id="c-new", doc_id="d-1", text="新数据",
            chunk_index=1,
        )
        ctx = PipelineContext(document=doc, chunks=[pre_embedded, new_chunk])

        result = await stage.run(ctx)

        assert result.chunks[0].embedding == [0.5] * 1024
        assert result.chunks[1].embedding is not None
        assert model.encode.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_chunks_skips(self):
        """chunks 为空时跳过，不报错"""
        model = _make_mock_embedding_model()
        stage = EmbedderStage(embedding_model=model)

        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        ctx = PipelineContext(document=doc, chunks=[])

        result = await stage.run(ctx)
        assert result.chunks == []
        model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_metadata(self):
        model = _make_mock_embedding_model()
        stage = EmbedderStage(embedding_model=model)

        doc = Document(doc_id="d-1", source_path=Path("test.md"), file_type="md")
        chunks = [
            Chunk(chunk_id="c-1", doc_id="d-1", text="A", chunk_index=0),
        ]
        ctx = PipelineContext(document=doc, chunks=chunks)

        result = await stage.run(ctx)
        assert "embedding_batches" in result.metadata
        assert result.metadata["embedding_batches"] >= 1

    @pytest.mark.asyncio
    async def test_name_and_fatal(self):
        stage = EmbedderStage(embedding_model=_make_mock_embedding_model())
        assert stage.name == "embedder"
        assert stage.fatal is False
