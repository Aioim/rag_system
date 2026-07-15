"""FAISSStore 测试"""
import numpy as np
import pytest

from config import settings
from retrieval.store import FAISSStore, get_store, reset_stores
from tests.unit.retrieval.conftest import make_chunk, one_hot, write_chunks


class TestFAISSStore:
    def test_missing_collection_raises(self, faiss_env):
        with pytest.raises(ValueError, match="nonexistent"):
            get_store("nonexistent")

    def test_get_chunk_builds_chunk(self, faiss_env):
        write_chunks([
            make_chunk(0, "文本零", one_hot(0), prev_id=None, next_id="c1"),
            make_chunk(1, "文本一", one_hot(1), prev_id="c0"),
        ])
        store = get_store("test")
        c = store.get_chunk("c0")
        assert c is not None
        assert c.text == "文本零"
        assert c.doc_id == "d1"
        assert c.next_chunk_id == "c1"
        assert store.get_chunk("missing") is None
        # 每次返回新实例（防调用方污染 docstore）
        assert store.get_chunk("c0") is not c

    def test_search_returns_ranked_chunk_ids(self, faiss_env):
        write_chunks([make_chunk(i, f"t{i}", one_hot(i)) for i in range(3)])
        store = get_store("test")
        q = np.array(one_hot(1), dtype=np.float32)
        result = store.search(q, k=2)
        assert result[0] == "c1"
        assert len(result) == 2

    def test_search_k_exceeds_ntotal(self, faiss_env):
        write_chunks([make_chunk(0, "t0", one_hot(0))])
        store = get_store("test")
        q = np.array(one_hot(0), dtype=np.float32)
        assert store.search(q, k=10) == ["c0"]

    def test_reconstruct_returns_vector(self, faiss_env):
        write_chunks([make_chunk(0, "t0", one_hot(0))])
        store = get_store("test")
        vec = store.reconstruct("c0")
        assert vec is not None
        # COSINE 写入时已归一化，one-hot 归一化后不变
        np.testing.assert_allclose(vec, np.array(one_hot(0), dtype=np.float32))
        assert store.reconstruct("missing") is None

    def test_all_chunks(self, faiss_env):
        write_chunks([make_chunk(i, f"t{i}", one_hot(i)) for i in range(3)])
        store = get_store("test")
        pairs = store.all_chunks()
        assert ("c0", "t0") in pairs
        assert len(pairs) == 3

    def test_is_empty_and_reload_bumps_version(self, faiss_env):
        write_chunks([make_chunk(0, "t0", one_hot(0))])
        store = get_store("test")
        assert not store.is_empty
        v0 = store.version
        # 追加写入后 reload 可见新数据
        write_chunks([make_chunk(1, "t1", one_hot(1))])
        store.reload()
        assert store.version == v0 + 1
        assert store.get_chunk("c1") is not None

    def test_ivf_index_load(self, faiss_env):
        """IVF 分支：nprobe 设置 + direct map 重建可用"""
        settings.faiss.index_type = "IVF_FLAT"
        write_chunks([make_chunk(i, f"t{i}", one_hot(i % 8, scale=1.0 + i))
                      for i in range(12)])
        store = get_store("test")
        q = np.array(one_hot(3), dtype=np.float32)
        assert len(store.search(q, k=2)) == 2
        assert store.reconstruct("c3") is not None

    def test_get_store_caches_instance(self, faiss_env):
        write_chunks([make_chunk(0, "t0", one_hot(0))])
        assert get_store("test") is get_store("test")
        reset_stores()
        # 重置后是新实例
        assert get_store("test") is not None
