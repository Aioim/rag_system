"""FactChecker 测试"""
import json

import pytest

from generation.fact_checker import FactChecker, FactCheckResult

from .conftest import MockLLM

CONTEXT = "[1] 年假为每年5天。\n\n[2] 病假需提供医院证明。"


def _llm_with_claims(claims: list[dict]) -> MockLLM:
    return MockLLM(response=json.dumps(claims, ensure_ascii=False))


class TestCheck:
    async def test_all_supported(self):
        llm = _llm_with_claims([
            {"claim": "年假为每年5天", "status": "supported"},
            {"claim": "病假需要医院证明", "status": "supported"},
        ])
        checker = FactChecker(llm)

        results, pass_rate, degraded = await checker.check("年假5天，病假要证明", CONTEXT)

        assert len(results) == 2
        assert all(r.status == "supported" for r in results)
        assert pass_rate == 1.0
        assert degraded is False

    async def test_partial_unsupported(self):
        llm = _llm_with_claims([
            {"claim": "年假为每年5天", "status": "supported"},
            {"claim": "加班有三倍工资", "status": "unsupported"},
        ])
        checker = FactChecker(llm)

        results, pass_rate, degraded = await checker.check("答案", CONTEXT)

        assert results[1].status == "unsupported"
        assert pass_rate == 0.5
        assert degraded is False

    async def test_contradicted(self):
        llm = _llm_with_claims([
            {"claim": "年假为每年10天", "status": "contradicted"},
        ])
        checker = FactChecker(llm)

        results, pass_rate, degraded = await checker.check("答案", CONTEXT)

        assert results[0].status == "contradicted"
        assert pass_rate == 0.0
        assert degraded is False

    async def test_invalid_status_normalized_to_unsupported(self):
        """LLM 返回未知 status 时按 unsupported 处理，不崩溃"""
        llm = _llm_with_claims([{"claim": "断言", "status": "weird"}])
        checker = FactChecker(llm)

        results, _, degraded = await checker.check("答案", CONTEXT)

        assert results[0].status == "unsupported"
        assert degraded is False

    async def test_temperature_zero_passed(self):
        llm = _llm_with_claims([{"claim": "a", "status": "supported"}])
        checker = FactChecker(llm)

        await checker.check("答案", CONTEXT)

        _, kwargs = llm.calls[0]
        assert kwargs.get("temperature") == 0


class TestDegradation:
    """核查失败不阻塞答案返回"""

    async def test_llm_failure_returns_empty_and_full_pass(self):
        checker = FactChecker(MockLLM(should_fail=True))

        results, pass_rate, degraded = await checker.check("答案", CONTEXT)

        assert results == []
        assert pass_rate == 1.0
        assert degraded is True

    async def test_json_parse_failure_degrades(self):
        checker = FactChecker(MockLLM(response="这不是JSON"))

        results, pass_rate, degraded = await checker.check("答案", CONTEXT)

        assert results == []
        assert pass_rate == 1.0
        assert degraded is True

    async def test_empty_answer_skips_llm(self):
        llm = MockLLM()
        checker = FactChecker(llm)

        results, pass_rate, degraded = await checker.check("", CONTEXT)

        assert results == []
        assert pass_rate == 1.0
        assert degraded is False
        assert llm.calls == []

    async def test_empty_context_skips_llm(self):
        llm = MockLLM()
        checker = FactChecker(llm)

        results, pass_rate, degraded = await checker.check("答案", "")

        assert results == []
        assert pass_rate == 1.0
        assert degraded is False
        assert llm.calls == []


class TestInjectWarnings:
    @pytest.fixture
    def checker(self):
        return FactChecker(MockLLM())

    def test_no_issues_returns_answer_unchanged(self, checker):
        results = [FactCheckResult(claim="a", status="supported")]
        assert checker.inject_warnings("原始答案", results) == "原始答案"

    def test_unsupported_appends_warning(self, checker):
        results = [FactCheckResult(claim="加班有三倍工资", status="unsupported")]

        annotated = checker.inject_warnings("原始答案", results)

        assert annotated.startswith("原始答案")
        assert "加班有三倍工资" in annotated
        assert "未在参考资料中找到依据" in annotated

    def test_contradicted_appends_conflict_warning(self, checker):
        results = [FactCheckResult(claim="年假为每年10天", status="contradicted")]

        annotated = checker.inject_warnings("原始答案", results)

        assert "年假为每年10天" in annotated
        assert "与参考资料冲突" in annotated

    def test_empty_results_returns_answer(self, checker):
        assert checker.inject_warnings("答案", []) == "答案"
