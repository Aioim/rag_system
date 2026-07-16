"""Reranker（CrossEncoder 精排 + MMR）测试"""
import numpy as np

from models.chunk import Chunk
from retrieval.reranker import Reranker, mmr_select


def _chunk(cid: str, text: str = "", score: float = 0.0) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", text=text, chunk_index=0,
                 rerank_score=score)


class MockCrossEncoder:
    """含"年假"的文本高分，其余低分"""

    def predict(self, pairs):
        return np.array(
            [0.9 if "年假" in text else 0.2 for _, text in pairs]
        )


class TestReranker:
    def test_scores_written_and_sorted(self):
        chunks = [_chunk("c0", "报销流程"), _chunk("c1", "年假申请")]
        result = Reranker(MockCrossEncoder()).rerank("年假", chunks)
        assert [c.chunk_id for c in result] == ["c1", "c0"]
        assert result[0].rerank_score == 0.9
        assert result[1].rerank_score == 0.2

    def test_empty_chunks(self):
        assert Reranker(MockCrossEncoder()).rerank("q", []) == []


class TestMMRSelect:
    # a/b 向量相同（冗余），c 正交（多样）
    VECTORS = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([1.0, 0.0], dtype=np.float32),
        "c": np.array([0.0, 1.0], dtype=np.float32),
    }

    def _chunks(self):
        return [
            _chunk("a", score=0.9),
            _chunk("b", score=0.8),
            _chunk("c", score=0.7),
        ]

    def test_lambda_1_pure_relevance(self):
        result = mmr_select(self._chunks(), self.VECTORS, top_k=2, mmr_lambda=1.0)
        assert [c.chunk_id for c in result] == ["a", "b"]

    def test_lambda_0_pure_diversity(self):
        # 首选最高分 a；之后 b 与 a 相似度 1、c 与 a 相似度 0 → 选 c
        result = mmr_select(self._chunks(), self.VECTORS, top_k=2, mmr_lambda=0.0)
        assert [c.chunk_id for c in result] == ["a", "c"]

    def test_pool_smaller_than_top_k(self):
        chunks = self._chunks()
        result = mmr_select(chunks, self.VECTORS, top_k=10, mmr_lambda=0.7)
        assert len(result) == 3
        assert result[0].chunk_id == "a"

    def test_missing_vector_treated_as_diverse(self):
        vectors = {"a": np.array([1.0, 0.0], dtype=np.float32),
                   "b": None, "c": None}
        result = mmr_select(self._chunks(), vectors, top_k=2, mmr_lambda=0.0)
        # b 向量缺失 → 相似度按 0 处理，仍按 MMR 得分参与竞争（0 - 0 > 0 - sim(c)? 均为 0，取先者 b）
        assert result[0].chunk_id == "a"
        assert len(result) == 2

    def test_empty(self):
        assert mmr_select([], {}, top_k=5, mmr_lambda=0.7) == []
