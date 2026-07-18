"""CitationBuilder 测试"""
from generation.citation_builder import CitationBuilder
from models.api import Source

from .conftest import make_chunk


class TestCitationBuilder:
    def test_builds_sources_from_chunks(self):
        chunks = [
            make_chunk(
                "c1", "年假申请流程", 0.92,
                doc_id="doc-a", metadata={"doc_title": "员工手册"},
            ),
            make_chunk(
                "c2", "病假条款", 0.81,
                doc_id="doc-b", metadata={"doc_title": "考勤制度"},
            ),
        ]

        sources = CitationBuilder.build(chunks)

        assert len(sources) == 2
        assert all(isinstance(s, Source) for s in sources)
        assert sources[0].doc_id == "doc-a"
        assert sources[0].doc_title == "员工手册"
        assert sources[0].chunk_text == "年假申请流程"
        assert sources[0].score == 0.92

    def test_doc_title_falls_back_to_doc_id(self):
        chunks = [make_chunk("c1", "文本", 0.5, doc_id="doc-x", metadata={})]

        sources = CitationBuilder.build(chunks)

        assert sources[0].doc_title == "doc-x"

    def test_preserves_chunk_order(self):
        chunks = [
            make_chunk("c1", "第一", 0.3),
            make_chunk("c2", "第二", 0.9),
            make_chunk("c3", "第三", 0.6),
        ]

        sources = CitationBuilder.build(chunks)

        assert [s.chunk_text for s in sources] == ["第一", "第二", "第三"]

    def test_empty_chunks_returns_empty_list(self):
        assert CitationBuilder.build([]) == []
