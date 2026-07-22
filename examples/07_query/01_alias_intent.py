"""
01_alias_intent.py — 查询理解：别名映射与意图分类

演示内容：
  1. 别名映射 — 用户术语 → 标准术语（resolve_alias / resolve_aliases_in_text）
  2. 上下文消歧 — 多义词处理（alias_manager.get_candidates）
  3. 意图分类（IntentClassifier）+ 清晰度判断

运行方式：
  cd rag0709
  python examples/07_query/01_alias_intent.py

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
    print(f"  API Base: {settings.llm.api_base_url or 'https://api.deepseek.com/v1'}")

    # ── 2. 别名映射（独立使用） ─────────────────────────────────
    banner("2. 别名映射 — 用户术语 → 标准术语")

    from query.aliases import resolve_alias, resolve_aliases_in_text, alias_manager

    test_aliases = ["工资条", "公积金", "年假", "vpn", "入职"]
    print("  单个别名解析:")
    for alias in test_aliases:
        resolved = resolve_alias(alias)
        if resolved != alias:
            print(f"    {alias!r} → {resolved!r}")
        else:
            print(f"    {alias!r} → (保留原词)")

    # 全文本别名替换
    text = "我的工资单和公积金怎么查？年假还剩几天？"
    resolved_text = resolve_aliases_in_text(text)
    print(f"\n  全文本替换:")
    print(f"    输入: {text}")
    print(f"    输出: {resolved_text}")

    # ── 3. 上下文消歧 — 多义词 ──────────────────────────────────
    banner("3. 上下文消歧 — 多义词处理")

    candidates = alias_manager.get_candidates("系统")
    print(f"  '系统' 的候选含义: {candidates}")

    r_it = resolve_alias("系统", context_text="IT系统登录不上")
    print(f"  resolve_alias('系统', context='IT系统登录不上') → {r_it!r}")

    r_att = resolve_alias("系统", context_text="考勤系统打卡异常")
    print(f"  resolve_alias('系统', context='考勤系统打卡异常') → {r_att!r}")

    r_none = resolve_alias("系统")
    print(f"  resolve_alias('系统', context=None) → {r_none!r}  (歧义，保留原词)")

    # ── 4. 意图分类 ─────────────────────────────────────────────
    banner("4. 意图分类 + 清晰度判断")

    from query.intent_classifier import IntentClassifier

    classifier = IntentClassifier(llm)
    print("  组件温度: 0 (确定性输出)")

    test_queries = [
        "什么是带薪年休假？",
        "申请年假需要什么材料？",
        "年假和病假有什么区别？",
        "HR部门的电话是多少？",
        "帮帮我",
    ]

    for q in test_queries:
        result = await classifier.classify(q)
        print(f"\n  查询: {q!r}")
        print(f"    意图: {result.intent}")
        print(f"    清晰: {result.is_clear}")
        if result.clarification_question:
            print(f"    澄清问题: {result.clarification_question}")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 别名映射与意图分类演示完成")
    print()
    print("  下一步: 02_fusion_rewrite.py — 上下文融合与查询改写")


if __name__ == "__main__":
    asyncio.run(main())
