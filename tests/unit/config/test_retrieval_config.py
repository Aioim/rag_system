"""RetrievalConfig 新增字段测试"""
import pytest
from pydantic import ValidationError

from config import settings
from config.settings import RetrievalConfig


def test_max_rerank_candidates_default():
    assert settings.retrieval.max_rerank_candidates == 30


class TestFieldConstraints:
    """非法配置在加载时被拒绝，而非运行时崩溃/静默出错"""

    def test_top_k_rejects_non_positive(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=0)
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=-5)

    def test_rrf_k_rejects_non_positive(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(rrf_k=-1)
        with pytest.raises(ValidationError):
            RetrievalConfig(rrf_k=0)

    def test_mmr_lambda_rejects_out_of_range(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(mmr_lambda=2.0)
        with pytest.raises(ValidationError):
            RetrievalConfig(mmr_lambda=-0.1)

    def test_mmr_lambda_accepts_boundaries(self):
        assert RetrievalConfig(mmr_lambda=0.0).mmr_lambda == 0.0
        assert RetrievalConfig(mmr_lambda=1.0).mmr_lambda == 1.0

    def test_threshold_order_rejected_when_inverted(self):
        with pytest.raises(ValidationError, match="relevance_threshold_sufficient"):
            RetrievalConfig(
                relevance_threshold_sufficient=0.3,
                relevance_threshold_need_more=0.5,
            )
