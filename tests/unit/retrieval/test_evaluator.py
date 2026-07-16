"""Self-RAG 自评测试（默认阈值 sufficient=0.5 / need_more=0.3）"""
from models.chunk import Chunk
from models.enums import RetrievalEval
from retrieval.evaluator import evaluate


def _chunks(scores: list[float]) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", doc_id="d", text="t", chunk_index=i,
              rerank_score=s)
        for i, s in enumerate(scores)
    ]


class TestEvaluate:
    def test_sufficient(self):
        assert evaluate(_chunks([0.8, 0.6])) == RetrievalEval.SUFFICIENT

    def test_sufficient_boundary(self):
        assert evaluate(_chunks([0.5])) == RetrievalEval.SUFFICIENT

    def test_need_more(self):
        assert evaluate(_chunks([0.4, 0.4])) == RetrievalEval.NEED_MORE

    def test_need_more_boundary(self):
        assert evaluate(_chunks([0.3])) == RetrievalEval.NEED_MORE

    def test_insufficient(self):
        assert evaluate(_chunks([0.1, 0.2])) == RetrievalEval.INSUFFICIENT

    def test_empty_is_insufficient(self):
        assert evaluate([]) == RetrievalEval.INSUFFICIENT
