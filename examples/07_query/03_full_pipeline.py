"""
03_full_pipeline.py — 查询理解：完整流程与边界情况

演示内容：
  1. QueryUnderstandingLayer 完整流程
  2. 模糊问题短路
  3. 温度约定总结

运行方式：
  cd rag0709
  python examples/07_query/03_full_pipeline.py

前置条件：需在 .env 中配置 LLM_API_KEY
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main():
    from examples._llm import create_llm

    # ── 1. 初始化 LLM ───────────────────────────────────────────
    banner("1. 初始化 LLM")
    llm = create_llm(temperature=0)
    print(f"  ✅ LLM 已连接: {settings.llm.default}")

    # ── 2. QueryUnderstandingLayer 完整流程 ─────────────────────
    banner("2. QueryUnderstandingLayer 完整流程")

    from query import get_query_layer, reset_query_layer

    layer = get_query_layer(llm)
    ctx = await layer.process("什么是带薪年休假？")

    print(f"  输入: '什么是带薪年休假？'")
    print(f"  意图: {ctx.intent.value if ctx.intent else 'N/A'}")
    print(f"  改写查询: {ctx.rewritten_queries}")
    print(f"  需要澄清: {ctx.needs_clarification}")
    print(f"  Pipeline 流程: 别名映射 → 意图分类 → 上下文融合 → 查询改写")

    # ── 3. 模糊问题短路 ─────────────────────────────────────────
    banner("3. 模糊问题短路")

    ctx2 = await layer.process("帮帮我")
    print(f"  输入: '帮帮我'")
    print(f"  需要澄清: {ctx2.needs_clarification}")
    if ctx2.clarification_question:
        print(f"  澄清问题: {ctx2.clarification_question!r}")
    print(f"  说明: 高模糊问题直接返回澄清请求，不进入后续检索/生成")

    # ── 清理 ────────────────────────────────────────────────────
    reset_query_layer()

    # ── 4. 温度约定 ─────────────────────────────────────────────
    banner("4. 组件温度约定")

    print("  IntentClassifier  (t=0)   确定性意图分类")
    print("  ContextFuser      (t=0)   确定性指代消解")
    print("  KeywordRewriter   (t=0)   幂等关键词提取")
    print("  HyDERewriter      (t=0.3) 受控假设答案")
    print("  SynonymRewriter   (t=0.3) 多样性同义变体")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 查询理解完整流程演示完成")
    print()
    print("  编程接口:")
    print("    from query import get_query_layer")
    print("    layer = get_query_layer(llm)")
    print("    ctx = await layer.process('查询', session_id='可选')")


if __name__ == "__main__":
    asyncio.run(main())
