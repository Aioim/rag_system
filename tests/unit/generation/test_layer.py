"""GenerationLayer 主编排器测试"""
import pytest

from generation.layer import GenerationLayer
from models.enums import Intent, RetrievalEval

from .conftest import MockLLM, make_chunk


class TestSufficient:
    async def test_full_pipeline_populates_fields(self, sample_ctx):
        llm = MockLLM(response="RAG是检索增强生成的架构。")
        layer = GenerationLayer(llm)
        sample_ctx.intent = Intent.CONCEPT
        sample_ctx.retrieval_eval = RetrievalEval.SUFFICIENT

        result = await layer.generate(sample_ctx)

        assert result.answer == "RAG是检索增强生成的架构。"
        assert result.assembled_prompt != ""
        assert result.sources
        assert 0 <= result.confidence <= 1
        assert "generation_ms" in result.metadata

    async def test_intent_is_used_in_prompt(self, sample_ctx):
        llm = MockLLM()
        layer = GenerationLayer(llm)
        sample_ctx.intent = Intent.PROCEDURE

        await layer.generate(sample_ctx)

        prompt = llm.calls[0][0]
        assert "过程" in prompt or "步骤" in prompt or "流程" in prompt


class TestNeedMore:
    async def test_generates_but_marks_partial(self, sample_ctx):
        llm = MockLLM(response="部分回答")
        layer = GenerationLayer(llm)
        sample_ctx.retrieval_eval = RetrievalEval.NEED_MORE

        result = await layer.generate(sample_ctx)

        assert result.answer == "部分回答"
        assert result.fallback_level == "partial"
        assert not result.is_fallback


class TestInsufficient:
    async def test_short_circuits_without_calling_llm(self, sample_ctx):
        llm = MockLLM()
        layer = GenerationLayer(llm)
        sample_ctx.retrieval_eval = RetrievalEval.INSUFFICIENT

        result = await layer.generate(sample_ctx)

        assert result.answer == ""
        assert result.is_fallback is True
        assert result.fallback_level == "no_answer"
        assert result.sources == []
        assert result.confidence == 0.0
        assert llm.calls == []


class TestDegradation:
    async def test_llm_failure_still_returns_ctx(self, sample_ctx):
        """LLM 失败时不上抛异常，answer="" 但 ctx 正常返回"""
        layer = GenerationLayer(MockLLM(should_fail=True))
        result = await layer.generate(sample_ctx)

        assert result.answer == ""

    async def test_confidence_in_reasonable_range(self, sample_ctx):
        """正常生成后置信度落在合理范围内（rerank_avg=0.7, pass_rate 取决于核查）"""
        llm = MockLLM(response="一个回答")
        layer = GenerationLayer(llm)
        sample_ctx.reranked = [make_chunk("c1", "资料", 0.7)]

        result = await layer.generate(sample_ctx)

        # rerank_avg=0.7, pass_rate in [0, 1] -> confidence in [0.28, 0.82]
        assert 0.2 <= result.confidence <= 0.85


class TestConfidence:
    async def test_formula(self, sample_ctx):
        """confidence = 0.6 * avg_rerank + 0.4 * pass_rate

        FactChecker 的 MockLLM 返回非 JSON（"回答"）→ 解析失败 → degraded → *0.8
        """
        llm = MockLLM(response="回答")
        layer = GenerationLayer(llm)
        sample_ctx.reranked = [
            make_chunk("c1", "a", 0.8),
            make_chunk("c2", "b", 0.6),
        ]

        result = await layer.generate(sample_ctx)

        # 0.6*0.7 + 0.4*1.0 = 0.82; FactChecker 降级 → *0.8 = 0.656
        assert result.confidence == 0.656

    async def test_no_chunks_zero_confidence(self, sample_ctx):
        llm = MockLLM(response="回答")
        layer = GenerationLayer(llm)
        sample_ctx.reranked = []

        result = await layer.generate(sample_ctx)

        # avg_rerank=0.0, fact_check 因 context 为空跳过 → pass_rate=1.0
        assert result.confidence == 0.4


class TestMetadata:
    async def test_records_generation_duration(self, sample_ctx):
        llm = MockLLM(response="回答")
        layer = GenerationLayer(llm)
        result = await layer.generate(sample_ctx)

        assert "generation_ms" in result.metadata
        assert result.metadata["generation_ms"] >= 0
