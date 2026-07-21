"""
demo_generation.py — 生成层模块演示

演示内容：
  1. PromptAssembler — 上下文去重/截断/拼接
  2. LLMRouter — 意图路由 + 温度选取
  3. GenerationLayer — 完整生成流程
  4. FactChecker — 断言拆解 + 逐条核查
  5. CitationBuilder — 引用来源构建
  6. INSUFFICIENT / NEED_MORE 场景处理

运行方式：
  cd rag0709
  python examples/09_generation/demo_generation.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402, F401
_ = settings.env  # 强制触发 initialize()  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main():
    from types import SimpleNamespace

    # ── 1. Mock LLM ─────────────────────────────────────────────
    banner("1. 初始化 Mock LLM")

    class MockLLM:
        """Mock LLM — 区分 lightweight 和 default 模型"""

        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        async def ainvoke(self, prompt: str, **kwargs):
            self.calls.append((prompt[:200], kwargs))
            model = kwargs.get("model", "default")

            # 事实核查 — 返回断言列表
            if "核查" in prompt or "claim" in prompt.lower() or "assertion" in prompt.lower():
                return SimpleNamespace(content='''[
                    {"claim": "年假申请需要填写OA申请单", "status": "supported"},
                    {"claim": "审批由部门主管负责", "status": "supported"},
                    {"claim": "提交后1小时内完成审批", "status": "unsupported"},
                    {"claim": "年假可以无限累积", "status": "contradicted"}
                ]''')

            # 正常生成回答
            return SimpleNamespace(
                content="""带薪年休假（Paid Annual Leave）是员工依法享有的福利假期。

根据公司《员工手册》第三章第二节规定：
1. 工作满1年不满10年的员工，每年享受5天带薪年假
2. 工作满10年不满20年的员工，每年享受10天带薪年假

申请流程：
1. 登录OA系统 → 2. 填写年假申请单 → 3. 选择休假日期 → 4. 提交部门主管审批 → 5. HR确认"""
            )

    llm = MockLLM()
    print("  ✅ MockLLM 已创建")

    # ── 2. PromptAssembler 演示 ─────────────────────────────────
    banner("2. PromptAssembler — 上下文组装")

    from generation.prompt_assembler import PromptAssembler
    from models.chunk import Chunk

    assembler = PromptAssembler()

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1",
              text="带薪年休假是员工依法享有的假期。工作满1年可享受5天带薪年假。",
              chunk_index=0, rerank_score=0.95,
              metadata={"source": "员工手册", "section": "3.2"}),
        Chunk(chunk_id="c2", doc_id="d1",
              text="申请年假需要在OA系统中提交申请，经主管审批后由HR确认。",
              chunk_index=1, rerank_score=0.88,
              metadata={"source": "员工手册", "section": "3.3"}),
        Chunk(chunk_id="c3", doc_id="d1",
              text="员工手册包含了公司的各项规章制度，共有12个章节。",
              chunk_index=2, rerank_score=0.75,
              metadata={"source": "员工手册", "section": "1.1"}),
    ]

    assembled_prompt = assembler.assemble(chunks)
    print(f"  输入 chunk 数: {len(chunks)}")
    print(f"  去重阈值:      {settings.generation.dedup_threshold}")
    print(f"  上下文预算:    {settings.generation.max_context_chars} 字符")
    print(f"  组装后长度:    {len(assembled_prompt)} 字符")
    print(f"\n  组装后 Prompt 预览:")
    print(f"  ---")
    print(f"  {assembled_prompt[:300]}...")
    print(f"  ---")

    # ── 3. LLMRouter 演示 ───────────────────────────────────────
    banner("3. LLMRouter — 意图路由 + 温度选取")

    from generation.llm_router import LLMRouter
    from models.enums import Intent

    router = LLMRouter()

    test_cases = [
        (Intent.CONCEPT, "什么是年假？"),
        (Intent.PROCEDURE, "年假怎么申请？"),
        (Intent.COMPARE, "年假和病假有什么区别？"),
        (Intent.LOOKUP, "HR电话是多少？"),
    ]

    print("  路由规则 (纯规则):")
    for intent, query in test_cases:
        result = router.route(intent)
        print(f"    {intent.value:10s} → model={result.model_name:20s}  tier={result.model_tier:12s} t={result.temperature}")

    print(f"\n  各意图温度配置:")
    for k, v in settings.llm.temperatures.items():
        print(f"    {k}: {v}")

    # ── 4. GenerationLayer 完整流程 ─────────────────────────────
    banner("4. GenerationLayer — 完整生成流程")

    from generation import get_generation_layer, reset_generation_layer
    from models.context import PipelineContext
    from models.enums import RetrievalEval

    layer = get_generation_layer(llm)

    # SUFFICIENT 场景
    ctx = PipelineContext(query="什么是带薪年休假？")
    ctx.intent = Intent.CONCEPT
    ctx.retrieval_eval = RetrievalEval.SUFFICIENT
    ctx.reranked = chunks
    ctx.assembled_prompt = assembled_prompt
    ctx = await layer.generate(ctx)

    print(f"  场景: SUFFICIENT (检索充分)")
    print(f"  回答: {ctx.answer[:150]}...")
    print(f"  置信度: {ctx.confidence}")
    print(f"  来源数: {len(ctx.sources)}")

    # NEED_MORE 场景
    ctx2 = PipelineContext(query="年假申请需要什么材料？")
    ctx2.intent = Intent.PROCEDURE
    ctx2.retrieval_eval = RetrievalEval.NEED_MORE
    ctx2.reranked = [
        Chunk(chunk_id="c_weak", doc_id="d1",
              text="员工应遵守公司各项规章制度。",
              chunk_index=0, rerank_score=0.45,
              metadata={"source": "员工手册"}),
    ]
    ctx2.assembled_prompt = assembler.assemble(ctx2.reranked)
    ctx2 = await layer.generate(ctx2)
    print(f"\n  场景: NEED_MORE (资料不足)")
    print(f"  回答: {ctx2.answer[:120]}...")

    # INSUFFICIENT 场景 — 短路
    ctx3 = PipelineContext(query="今天天气怎么样？")
    ctx3.retrieval_eval = RetrievalEval.INSUFFICIENT
    ctx3 = await layer.generate(ctx3)
    print(f"\n  场景: INSUFFICIENT (完全不足)")
    print(f"  回答: {ctx3.answer!r} (空，短路不调用 LLM)")
    print(f"  fallback_level: {ctx3.fallback_level.value}")

    # ── 5. FactChecker 演示 ─────────────────────────────────────
    banner("5. FactChecker — 断言拆解 + 逐条核查")

    from generation.fact_checker import FactChecker

    checker = FactChecker(llm)

    answer = """带薪年休假申请流程如下：
    1. 登录OA系统
    2. 填写年假申请单
    3. 选择休假日期
    4. 提交部门主管审批（1小时内完成）
    5. 年假可以无限累积
    """
    context_text = "年假申请流程：登录OA → 填写申请单 → 主管审批 → HR确认。年假最多累积至次年3月。"

    results, pass_rate, degraded = await checker.check(answer, context_text)

    print(f"  核查结果: {len(results)} 条断言, 通过率: {pass_rate:.0%}")
    for r in results:
        icon = {"supported": "✅", "unsupported": "⚠️", "contradicted": "❌"}.get(r.status, "?")
        print(f"    {icon} [{r.status}] {r.claim}")

    # ── 6. CitationBuilder 演示 ─────────────────────────────────
    banner("6. CitationBuilder — 引用来源构建")

    from generation.citation_builder import CitationBuilder

    builder = CitationBuilder()
    sources = builder.build(chunks)

    print(f"  构建 {len(sources)} 个引用来源:")
    for i, src in enumerate(sources, 1):
        print(f"    [{i}] {src.doc_title} (chunk={src.doc_id}, score={src.score})")
        print(f"        {src.chunk_text[:80]}...")

    # ── 7. 置信度计算 ───────────────────────────────────────────
    banner("7. 置信度计算")

    print(f"  公式: confidence = 0.6 * rerank_avg + 0.4 * fact_pass_rate")
    rerank_scores = [c.rerank_score for c in chunks if c.rerank_score]
    avg_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0
    confidence = 0.6 * avg_score + 0.4 * pass_rate
    print(f"  rerank_avg     = {avg_score:.3f}")
    print(f"  fact_pass_rate = {pass_rate:.3f}")
    print(f"  confidence     = {confidence:.3f}")

    # ── 清理 ────────────────────────────────────────────────────
    reset_generation_layer()

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 生成模块演示完成")
    print()
    print("  Pipeline 流程: 组装 → 路由 → 生成 → 核查 → 引用")
    print()
    print("  关键配置 (settings.generation):")
    print(f"    dedup_threshold:    0.85  上下文去重阈值")
    print(f"    max_context_chars:  9000  上下文预算")
    print(f"    fact_check_enabled: true  事实核查开关")


if __name__ == "__main__":
    asyncio.run(main())
