"""FAISSIndexWriter 测试"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ingestion.context import Chunk
from ingestion.indexer import FAISSIndexWriter


def _make_chunks(n: int = 5, dim: int = 128) -> list[Chunk]:
    """创建带随机 embedding 的测试 chunk"""
    chunks = []
    for i in range(n):
        emb = np.random.rand(dim).astype(np.float32).tolist()
        chunks.append(
            Chunk(
                chunk_id=f"c-{i:03d}",
                doc_id="doc-test",
                text=f"chunk text {i}",
                chunk_index=i,
                embedding=emb,
                metadata={"source": "test"},
            )
        )
    return chunks


class TestFAISSIndexWriter:
    def test_write_creates_index_and_docstore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss["index_dir"]
            original_dim = settings.faiss["dimension"]
            settings.faiss["index_dir"] = tmpdir
            settings.faiss["dimension"] = 128

            try:
                writer = FAISSIndexWriter()
                chunks = _make_chunks(5)
                writer.write(chunks, "test_collection")

                idx_dir = Path(tmpdir) / "test_collection"
                assert idx_dir.exists()
                assert (idx_dir / "index.faiss").exists()
                assert (idx_dir / "docstore.json").exists()
            finally:
                settings.faiss["index_dir"] = original_dir
                settings.faiss["dimension"] = original_dim

    def test_docstore_contains_chunk_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss["index_dir"]
            original_dim = settings.faiss["dimension"]
            settings.faiss["index_dir"] = tmpdir
            settings.faiss["dimension"] = 128

            try:
                writer = FAISSIndexWriter()
                chunks = _make_chunks(3)
                writer.write(chunks, "test_collection")

                docstore_path = Path(tmpdir) / "test_collection" / "docstore.json"
                with open(docstore_path, encoding="utf-8") as f:
                    docstore = json.load(f)

                assert "c-000" in docstore
                assert docstore["c-000"]["text"] == "chunk text 0"
                assert docstore["c-000"]["doc_id"] == "doc-test"
                assert "faiss_id" in docstore["c-000"]
            finally:
                settings.faiss["index_dir"] = original_dir
                settings.faiss["dimension"] = original_dim

    def test_dimension_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss["index_dir"]
            original_dim = settings.faiss["dimension"]
            settings.faiss["index_dir"] = tmpdir
            settings.faiss["dimension"] = 256

            try:
                writer = FAISSIndexWriter()
                chunks = _make_chunks(5, dim=128)
                with pytest.raises(ValueError, match="维度"):
                    writer.write(chunks, "test_collection")
            finally:
                settings.faiss["index_dir"] = original_dir
                settings.faiss["dimension"] = original_dim

    def test_append_to_existing_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss["index_dir"]
            original_dim = settings.faiss["dimension"]
            settings.faiss["index_dir"] = tmpdir
            settings.faiss["dimension"] = 128

            try:
                writer = FAISSIndexWriter()
                batch1 = _make_chunks(3)
                writer.write(batch1, "test_collection")
                # batch2 使用不同的 chunk_id 前缀避免覆盖
                batch2 = [
                    Chunk(chunk_id="c-new-0", doc_id="doc-test", text="new 0",
                          chunk_index=0, embedding=np.random.rand(128).astype(np.float32).tolist()),
                    Chunk(chunk_id="c-new-1", doc_id="doc-test", text="new 1",
                          chunk_index=1, embedding=np.random.rand(128).astype(np.float32).tolist()),
                ]
                writer.write(batch2, "test_collection")

                docstore_path = Path(tmpdir) / "test_collection" / "docstore.json"
                with open(docstore_path, encoding="utf-8") as f:
                    docstore = json.load(f)
                assert len(docstore) == 5
            finally:
                settings.faiss["index_dir"] = original_dir
                settings.faiss["dimension"] = original_dim
