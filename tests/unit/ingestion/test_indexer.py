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

            original_dir = settings.faiss.index_dir
            original_dim = settings.faiss.dimension
            settings.faiss.index_dir = Path(tmpdir)
            settings.faiss.dimension = 128

            try:
                writer = FAISSIndexWriter()
                chunks = _make_chunks(5)
                writer.write(chunks, "test_collection")

                idx_dir = Path(tmpdir) / "test_collection"
                assert idx_dir.exists()
                assert (idx_dir / "index.faiss").exists()
                assert (idx_dir / "docstore.json").exists()
            finally:
                settings.faiss.index_dir = original_dir
                settings.faiss.dimension = original_dim

    def test_docstore_contains_chunk_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss.index_dir
            original_dim = settings.faiss.dimension
            settings.faiss.index_dir = Path(tmpdir)
            settings.faiss.dimension = 128

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
                settings.faiss.index_dir = original_dir
                settings.faiss.dimension = original_dim

    def test_dimension_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss.index_dir
            original_dim = settings.faiss.dimension
            settings.faiss.index_dir = Path(tmpdir)
            settings.faiss.dimension = 256

            try:
                writer = FAISSIndexWriter()
                chunks = _make_chunks(5, dim=128)
                with pytest.raises(ValueError, match="维度"):
                    writer.write(chunks, "test_collection")
            finally:
                settings.faiss.index_dir = original_dir
                settings.faiss.dimension = original_dim

    def test_append_to_existing_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss.index_dir
            original_dim = settings.faiss.dimension
            settings.faiss.index_dir = Path(tmpdir)
            settings.faiss.dimension = 128

            try:
                writer = FAISSIndexWriter()
                batch1 = _make_chunks(3)
                writer.write(batch1, "test_collection")
                # batch2 为另一文档（不同 doc_id）→ 应累积而非替换
                batch2 = [
                    Chunk(chunk_id="c-new-0", doc_id="doc-other", text="new 0",
                          chunk_index=0, embedding=np.random.rand(128).astype(np.float32).tolist()),
                    Chunk(chunk_id="c-new-1", doc_id="doc-other", text="new 1",
                          chunk_index=1, embedding=np.random.rand(128).astype(np.float32).tolist()),
                ]
                writer.write(batch2, "test_collection")

                docstore_path = Path(tmpdir) / "test_collection" / "docstore.json"
                with open(docstore_path, encoding="utf-8") as f:
                    docstore = json.load(f)
                assert len(docstore) == 5
            finally:
                settings.faiss.index_dir = original_dir
                settings.faiss.dimension = original_dim

    def test_rewrite_same_document_replaces_old_vectors(self):
        """同一文档重复写入应替换旧向量，而非堆积孤儿向量（复现索引膨胀 bug）"""
        import faiss

        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss.index_dir
            original_dim = settings.faiss.dimension
            settings.faiss.index_dir = Path(tmpdir)
            settings.faiss.dimension = 128

            try:
                writer = FAISSIndexWriter()
                # 首次摄取：5 个 chunk
                writer.write(_make_chunks(5), "test_collection")
                # 重新摄取同一文档（chunk_id 变化，doc_id 相同）：3 个 chunk
                batch2 = [
                    Chunk(chunk_id=f"c-re-{i}", doc_id="doc-test", text=f"re {i}",
                          chunk_index=i,
                          embedding=np.random.rand(128).astype(np.float32).tolist())
                    for i in range(3)
                ]
                writer.write(batch2, "test_collection")

                idx_dir = Path(tmpdir) / "test_collection"
                index = faiss.read_index(str(idx_dir / "index.faiss"))
                with open(idx_dir / "docstore.json", encoding="utf-8") as f:
                    docstore = json.load(f)

                # 旧 5 条应被清理，只保留本次 3 条；索引向量数一致
                assert len(docstore) == 3
                assert set(docstore) == {"c-re-0", "c-re-1", "c-re-2"}
                assert index.ntotal == 3
                assert sorted(e["faiss_id"] for e in docstore.values()) == [0, 1, 2]
            finally:
                settings.faiss.index_dir = original_dir
                settings.faiss.dimension = original_dim

    def test_rewrite_preserves_other_documents(self):
        """重写某文档时，其他文档的向量应保留且 faiss_id 重映射后仍指向正确向量"""
        import faiss

        with tempfile.TemporaryDirectory() as tmpdir:
            from config import settings

            original_dir = settings.faiss.index_dir
            original_dim = settings.faiss.dimension
            settings.faiss.index_dir = Path(tmpdir)
            settings.faiss.dimension = 128

            try:
                writer = FAISSIndexWriter()
                writer.write(_make_chunks(3), "test_collection")  # doc-test

                # doc-b 使用确定性 one-hot 向量（归一化不变），便于校验重映射
                def _one_hot(pos: int) -> list[float]:
                    v = np.zeros(128, dtype=np.float32)
                    v[pos] = 1.0
                    return v.tolist()

                batch_b = [
                    Chunk(chunk_id=f"b-{i}", doc_id="doc-b", text=f"b {i}",
                          chunk_index=i, embedding=_one_hot(10 + i))
                    for i in range(2)
                ]
                writer.write(batch_b, "test_collection")

                # 重写 doc-test
                writer.write(_make_chunks(2), "test_collection")

                idx_dir = Path(tmpdir) / "test_collection"
                index = faiss.read_index(str(idx_dir / "index.faiss"))
                with open(idx_dir / "docstore.json", encoding="utf-8") as f:
                    docstore = json.load(f)

                # doc-b 2 条 + 重写后的 doc-test 2 条
                assert len(docstore) == 4
                assert index.ntotal == 4
                assert sorted(e["faiss_id"] for e in docstore.values()) == [0, 1, 2, 3]
                # doc-b 的 faiss_id 重映射后仍指向原向量
                for i in range(2):
                    fid = docstore[f"b-{i}"]["faiss_id"]
                    vec = index.reconstruct(int(fid))
                    assert vec[10 + i] == pytest.approx(1.0)
            finally:
                settings.faiss.index_dir = original_dir
                settings.faiss.dimension = original_dim

    @pytest.mark.parametrize("bad_collection", [
        "",
        "path/traversal",
        "path\\traversal",
        "../escape",
        "nested/../escape",
    ])
    def test_rejects_dangerous_collection_names(self, bad_collection):
        """非法 collection 名称应抛出 ValueError，防止路径遍历"""
        writer = FAISSIndexWriter()
        chunks = _make_chunks(1, dim=128)
        with pytest.raises(ValueError, match="非法 collection 名称"):
            writer.write(chunks, bad_collection)
