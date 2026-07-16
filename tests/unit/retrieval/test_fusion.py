"""RRF 融合测试"""
import pytest

from retrieval.fusion import rrf_fuse


class TestRRFFuse:
    def test_score_math(self):
        # a 在两路均排第 1：score = 2/(60+1)；b 两路第 2：2/(60+2)；c 单路第 3：1/(60+3)
        result = rrf_fuse([["a", "b", "c"], ["a", "b"]], rrf_k=60, limit=10)
        assert [cid for cid, _ in result] == ["a", "b", "c"]
        scores = dict(result)
        assert scores["a"] == pytest.approx(2 / 61)
        assert scores["b"] == pytest.approx(2 / 62)
        assert scores["c"] == pytest.approx(1 / 63)

    def test_dedup_across_lists(self):
        result = rrf_fuse([["a"], ["a"], ["a"]], rrf_k=60, limit=10)
        assert len(result) == 1

    def test_limit_truncates(self):
        result = rrf_fuse([["a", "b", "c", "d"]], rrf_k=60, limit=2)
        assert len(result) == 2
        assert result[0][0] == "a"

    def test_empty_input(self):
        assert rrf_fuse([], rrf_k=60, limit=5) == []
        assert rrf_fuse([[], []], rrf_k=60, limit=5) == []
