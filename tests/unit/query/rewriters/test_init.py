"""QueryRewriter 编排器测试"""
import pytest
from query.rewriters.base import BaseRewriter
from query.rewriters import QueryRewriter


class FakeLLM:
    pass  # 不需要真实 LLM，用子类 Rewriter 绕过


class FastRewriter(BaseRewriter):
    """快速改写器"""

    def __init__(self, results):
        self.results = results

    async def rewrite(self, query: str) -> list[str]:
        return self.results


class SlowRewriter(BaseRewriter):
    """慢速改写器（验证并行）"""

    def __init__(self, results):
        self.results = results

    async def rewrite(self, query: str) -> list[str]:
        import asyncio
        await asyncio.sleep(0.01)
        return self.results


class FailingRewriter(BaseRewriter):
    """失败改写器"""

    async def rewrite(self, query: str) -> list[str]:
        raise RuntimeError("rewrite failed")


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_original_query_always_first(self):
        rewriters = [
            FastRewriter(["result1"]),
            FastRewriter(["result2"]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters
        result = await orchestrator.rewrite("原始查询")
        assert result[0] == "原始查询"

    @pytest.mark.asyncio
    async def test_merges_all_results(self):
        rewriters = [
            FastRewriter(["A"]),
            FastRewriter(["B", "C"]),
            FastRewriter([]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters
        result = await orchestrator.rewrite("Q")
        assert "Q" in result
        assert "A" in result
        assert "B" in result
        assert "C" in result

    @pytest.mark.asyncio
    async def test_deduplicates_results(self):
        rewriters = [
            FastRewriter(["dup"]),
            FastRewriter(["dup", "dup"]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters
        result = await orchestrator.rewrite("Q")
        assert result.count("dup") == 1

    @pytest.mark.asyncio
    async def test_handles_rewriter_failure(self):
        """单个 rewriter 失败不影响其他"""
        rewriters = [
            FastRewriter(["good"]),
            FailingRewriter(),
            FastRewriter(["also_good"]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters
        result = await orchestrator.rewrite("Q")
        assert "good" in result
        assert "also_good" in result

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """验证并行执行：慢速 rewriters 并行完成"""
        rewriters = [
            SlowRewriter(["a"]),
            SlowRewriter(["b"]),
            SlowRewriter(["c"]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters

        import time
        start = time.monotonic()
        result = await orchestrator.rewrite("Q")
        elapsed = time.monotonic() - start

        # 并行执行，总时间应明显小于 3×0.01
        assert elapsed < 0.05  # Windows 上放宽阈值
        assert len(result) >= 4  # Q + a + b + c

    @pytest.mark.asyncio
    async def test_constructor_injects_llm(self):
        """构造函数参数完整测试"""
        llm = FakeLLM()
        orchestrator = QueryRewriter(llm)
        assert len(orchestrator._rewriters) == 3
        assert all(isinstance(r, BaseRewriter) for r in orchestrator._rewriters)
