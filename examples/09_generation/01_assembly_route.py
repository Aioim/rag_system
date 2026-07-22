"""
01_assembly_route.py — 生成层：Prompt 组装与模型路由

演示内容：
  1. PromptAssembler — 上下文去重/截断/拼接
  2. LLMRouter — 意图路由 + 温度选取（纯规则）

运行方式：
  cd rag0709
  python examples/09_generation/01_assembly_route.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. PromptAssembler 演示 ─────────────────────────────────
    banner("1. PromptAssembler — 上下文组装")

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

    # ── 2. LLMRouter 演示 ───────────────────────────────────────
    banner("2. LLMRouter — 意图路由 + 温度选取")

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

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 组装与路由演示完成")
    print()
    print("  下一步: 02_generate_factcheck.py — 完整生成流程与事实核查")


if __name__ == "__main__":
    main()
