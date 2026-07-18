"""PromptAssembler 测试"""
from generation.prompt_assembler import PromptAssembler

from .conftest import make_chunk


class TestDedup:
    def test_removes_similar_keeps_higher_score(self):
        """余弦相似度 > 阈值的 chunk 只保留高分者"""
        chunks = [
            make_chunk("c1", "年假规定A", 0.9, embedding=[1.0, 0.0]),
            make_chunk("c2", "年假规定B", 0.5, embedding=[0.999, 0.01]),
            make_chunk("c3", "报销流程", 0.7, embedding=[0.0, 1.0]),
        ]

        kept = PromptAssembler.dedup(chunks, threshold=0.85)

        kept_ids = [c.chunk_id for c in kept]
        assert "c1" in kept_ids
        assert "c2" not in kept_ids
        assert "c3" in kept_ids

    def test_falls_back_to_text_dedup_without_embedding(self):
        """embedding 缺失时降级为文本精确去重"""
        chunks = [
            make_chunk("c1", "相同文本", 0.9),
            make_chunk("c2", "相同文本", 0.5),
            make_chunk("c3", "不同文本", 0.7),
        ]

        kept = PromptAssembler.dedup(chunks, threshold=0.85)

        kept_ids = [c.chunk_id for c in kept]
        assert kept_ids == ["c1", "c3"]

    def test_preserves_ranking_order(self):
        """去重后保留原始排序（reranked 顺序即最终排名）"""
        chunks = [
            make_chunk("c1", "文本一", 0.9, embedding=[1.0, 0.0]),
            make_chunk("c2", "文本二", 0.8, embedding=[0.0, 1.0]),
            make_chunk("c3", "文本三", 0.7, embedding=[0.7, 0.7]),
        ]

        kept = PromptAssembler.dedup(chunks, threshold=0.99)

        assert [c.chunk_id for c in kept] == ["c1", "c2", "c3"]

    def test_empty_chunks(self):
        assert PromptAssembler.dedup([], threshold=0.85) == []


class TestAllocateBudget:
    def test_stops_when_budget_exhausted(self):
        chunks = [
            make_chunk("c1", "一" * 40, 0.9),
            make_chunk("c2", "二" * 40, 0.8),
            make_chunk("c3", "三" * 40, 0.7),
        ]

        kept = PromptAssembler.allocate_budget(chunks, max_chars=100)

        assert [c.chunk_id for c in kept] == ["c1", "c2"]

    def test_top1_always_included_and_truncated(self):
        """Top-1 超预算时截断保留，不丢弃（Lost-in-the-Middle 缓解：Top-1 置顶）"""
        chunks = [
            make_chunk("c1", "长" * 200, 0.9),
            make_chunk("c2", "短文本", 0.8),
        ]

        kept = PromptAssembler.allocate_budget(chunks, max_chars=100)

        assert len(kept) == 1
        assert kept[0].chunk_id == "c1"
        assert len(kept[0].text) == 100

    def test_all_fit_within_budget(self):
        chunks = [make_chunk("c1", "短", 0.9), make_chunk("c2", "文", 0.8)]

        kept = PromptAssembler.allocate_budget(chunks, max_chars=100)

        assert [c.chunk_id for c in kept] == ["c1", "c2"]

    def test_budget_truncation_does_not_mutate_input(self):
        """截断产生新 Chunk，不修改原对象"""
        original = make_chunk("c1", "长" * 200, 0.9)

        kept = PromptAssembler.allocate_budget([original], max_chars=100)

        assert len(original.text) == 200
        assert len(kept[0].text) == 100


class TestAssemble:
    def test_numbered_concat_with_top1_first(self):
        assembler = PromptAssembler()
        chunks = [
            make_chunk("c1", "第一段资料", 0.9, embedding=[1.0, 0.0]),
            make_chunk("c2", "第二段资料", 0.8, embedding=[0.0, 1.0]),
        ]

        result = assembler.assemble(chunks, max_chars=1000, threshold=0.85)

        assert result.index("[1] 第一段资料") < result.index("[2] 第二段资料")

    def test_empty_chunks_returns_empty_string(self):
        assert PromptAssembler().assemble([], max_chars=1000, threshold=0.85) == ""

    def test_defaults_read_from_settings(self):
        """不传参时使用 settings.generation 配置"""
        result = PromptAssembler().assemble([make_chunk("c1", "资料", 0.9)])
        assert "[1] 资料" in result
