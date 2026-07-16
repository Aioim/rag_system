"""VectorRetriever 测试"""
import numpy as np

from retrieval.vector_retriever import VectorRetriever


class FakeStore:
    def __init__(self, result):
        self.result = result
        self.last_vector = None
        self.last_k = None

    def search(self, vector, k):
        self.last_vector = vector
        self.last_k = k
        return self.result[:k]


class FakeEncoder:
    def encode(self, texts):
        # 固定返回 norm=2 的向量，验证 COSINE 归一化
        return np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)


class TestVectorRetriever:
    def test_retrieve_returns_store_result(self):
        store = FakeStore(["c1", "c0"])
        r = VectorRetriever(store, FakeEncoder())
        assert r.retrieve("查询", k=2) == ["c1", "c0"]
        assert store.last_k == 2

    def test_cosine_normalizes_query_vector(self):
        store = FakeStore(["c0"])
        r = VectorRetriever(store, FakeEncoder())
        r.retrieve("查询", k=1)
        # metric_type 默认 COSINE：norm=2 的向量应被归一化为单位向量
        np.testing.assert_allclose(
            store.last_vector, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
