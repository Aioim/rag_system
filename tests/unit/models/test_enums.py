"""Enums 测试"""
from models.enums import DocumentStatus, Intent, RetrievalEval


class TestIntent:
    def test_values(self):
        assert Intent.CONCEPT.value == "concept"
        assert Intent.PROCEDURE.value == "procedure"
        assert Intent.COMPARE.value == "compare"
        assert Intent.LOOKUP.value == "lookup"

    def test_from_string(self):
        assert Intent("concept") == Intent.CONCEPT


class TestRetrievalEval:
    def test_values(self):
        assert RetrievalEval.SUFFICIENT.value == "sufficient"
        assert RetrievalEval.NEED_MORE.value == "need_more"
        assert RetrievalEval.INSUFFICIENT.value == "insufficient"


class TestDocumentStatus:
    def test_values(self):
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.DONE.value == "done"
        assert DocumentStatus.FAILED.value == "failed"
