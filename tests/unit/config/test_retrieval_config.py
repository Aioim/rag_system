"""RetrievalConfig 新增字段测试"""
from config import settings


def test_max_rerank_candidates_default():
    assert settings.retrieval.max_rerank_candidates == 30
