"""BM25Retriever 测试（jieba 分词 + rank_bm25 内存索引）"""
from retrieval.bm25_retriever import BM25Retriever


class FakeStore:
    version = 0

    def __init__(self, pairs):
        self._pairs = pairs

    def all_chunks(self):
        return self._pairs


CORPUS = [
    ("c0", "申请年假需要提前三天提交审批"),
    ("c1", "薪资明细可在人事系统查询"),
    ("c2", "差旅报销需提供发票原件"),
]


class TestBM25Retriever:
    def test_chinese_term_match(self):
        r = BM25Retriever(FakeStore(CORPUS))
        result = r.retrieve("年假 审批", k=3)
        assert result[0] == "c0"

    def test_no_match_returns_empty(self):
        r = BM25Retriever(FakeStore(CORPUS))
        assert r.retrieve("量子计算", k=3) == []

    def test_k_truncates(self):
        r = BM25Retriever(FakeStore(CORPUS))
        result = r.retrieve("申请 查询 报销", k=1)
        assert len(result) == 1

    def test_empty_corpus(self):
        r = BM25Retriever(FakeStore([]))
        assert r.retrieve("年假", k=3) == []

    def test_records_store_version(self):
        store = FakeStore(CORPUS)
        store.version = 7
        assert BM25Retriever(store).version == 7
