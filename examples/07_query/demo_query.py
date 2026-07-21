"""
demo_query.py — 查询理解层模块演示

演示内容：
  1. 别名映射 — 用户术语 → 标准术语
  2. 上下文消歧 — 多义词处理
  3. 意图分类 — concept/procedure/compare/lookup
  4. 清晰度判断 — 模糊问题检测
  5. 多轮上下文融合 — 指代消解 + 追问补全
  6. 查询改写 — HyDE / 关键词 / 同义变体
  7. QueryUnderstandingLayer 完整流程

运行方式：
  cd rag0709
  python examples/07_query/demo_query.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402, F401
_ = settings.env  # 强制触发 initialize()，在导入其他模块前完成 _config 设置  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main():
    from types import SimpleNamespace

    # ── 1. Mock LLM ─────────────────────────────────────────────
    banner("1. 初始化 Mock LLM + QueryUnderstandingLayer")

    from config import settings

    class MockLLM:
        """可编程 Mock LLM — 不同 prompt 返回不同结果"""

        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        async def ainvoke(self, prompt: str, **kwargs):
            self.calls.append((prompt, kwargs))

            # 意图分类 (IntentClassifier)
            if "意图分类" in prompt or "is_clear" in prompt:
                if "帮帮我" in prompt:
                    return SimpleNamespace(content='{"intent": "concept", "is_clear": false, "clarification_question": "您想了解哪方面内容？"}')
                if "年假" in prompt and "材料" in prompt:
                    return SimpleNamespace(content='{"intent": "procedure", "is_clear": true, "clarification_question": null}')
                if "年假" in prompt and "病假" in prompt:
                    return SimpleNamespace(content='{"intent": "compare", "is_clear": true, "clarification_question": null}')
                if "什么是" in prompt:
                    return SimpleNamespace(content='{"intent": "concept", "is_clear": true, "clarification_question": null}')
                if "电话" in prompt or "地址" in prompt:
                    return SimpleNamespace(content='{"intent": "lookup", "is_clear": true, "clarification_question": null}')
                if "年假" in prompt:
                    return SimpleNamespace(content='{"intent": "procedure", "is_clear": true, "clarification_question": null}')
                return SimpleNamespace(content='{"intent": "concept", "is_clear": true, "clarification_question": null}')

            # 上下文融合 / 指代消解 (ContextFuser)
            if "上下文" in prompt or "补全" in prompt:
                if "材料" in prompt:
                    return SimpleNamespace(content='{"resolved_query": "申请年假需要什么材料？"}')
                if "上面" in prompt or "之前" in prompt:
                    return SimpleNamespace(content='{"resolved_query": "带薪年休假需要什么材料？"}')
                return SimpleNamespace(content='{"resolved_query": "已补全的查询"}')

            # HyDE 假设答案
            if "HyDE" in prompt or "假设答案" in prompt:
                return SimpleNamespace(content="假设的答案段落：RAG是检索增强生成（Retrieval-Augmented Generation）...")

            # 同义变体
            if "同义" in prompt or "synonym" in prompt.lower():
                return SimpleNamespace(content='["什么是RAG", "RAG技术介绍", "检索增强生成原理"]')

            # 关键词提取
            if "关键词" in prompt or "keyword" in prompt.lower() or "BM25" in prompt:
                return SimpleNamespace(content='["RAG", "检索增强生成", "知识库"]')

            # 默认
            return SimpleNamespace(content="默认响应")

    llm = MockLLM()
    print("  ✅ MockLLM 已创建（无需真实 LLM API）")

    # ── 2. 别名映射（独立使用） ─────────────────────────────────
    banner("2. 别名映射 — 用户术语 → 标准术语")

    from query.aliases import resolve_alias, resolve_aliases_in_text, alias_manager

    # 单个别名解析
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

    # "系统" 有多个含义
    candidates = alias_manager.get_candidates("系统")
    print(f"  '系统' 的候选含义: {candidates}")

    # IT 上下文
    r_it = resolve_alias("系统", context_text="IT系统登录不上")
    print(f"  resolve_alias('系统', context='IT系统登录不上') → {r_it!r}")

    # 考勤上下文
    r_att = resolve_alias("系统", context_text="考勤系统打卡异常")
    print(f"  resolve_alias('系统', context='考勤系统打卡异常') → {r_att!r}")

    # 无上下文 — 歧义时保留原词
    r_none = resolve_alias("系统")
    print(f"  resolve_alias('系统', context=None) → {r_none!r}  (歧义，保留原词)")

    # ── 4. 意图分类 ─────────────────────────────────────────────
    banner("4. 意图分类 + 清晰度判断")

    from query.intent_classifier import IntentClassifier

    classifier = IntentClassifier(llm)
    print("  组件温度: 0 (确定性输出)")

    # 明确问题
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

    # ── 5. 多轮上下文融合 ───────────────────────────────────────
    banner("5. 多轮上下文融合 — 指代消解 + 追问补全")

    from query.context_fuser import ContextFuser
    from models.session import Session, Message

    fuser = ContextFuser(llm)
    print("  组件温度: 0 (确定性输出)")

    # 模拟多轮对话的 Session
    session = Session(
        session_id="demo-session",
        messages=[
            Message(role="user", content="带薪年休假怎么申请？"),
            Message(role="assistant", content="申请带薪年休假需要提前在OA系统中提交申请，经主管审批。"),
        ]
    )

    # 追问补全
    fused = await fuser.fuse("需要什么材料？", session)
    print(f"  第1轮: 用户问'带薪年休假怎么申请？'")
    print(f"  第2轮: 用户问'需要什么材料？' (不完整)")
    print(f"  融合后: {fused!r}")
    print(f"  说明: 自动补全为完整的独立查询")

    # 指代消解 — 追加消息到 session
    session.messages.append(Message(role="user", content="需要什么材料？"))
    session.messages.append(Message(role="assistant", content="申请年假需要填写OA申请单、提供身份证明。"))
    fused2 = await fuser.fuse("上面提到的流程具体怎么做？", session)
    print(f"\n  第3轮: 用户问'上面提到的流程具体怎么做？'")
    print(f"  融合后: {fused2!r}")

    # ── 6. 查询改写 ─────────────────────────────────────────────
    banner("6. 查询改写 — HyDE / 关键词 / 同义变体")

    from query.rewriters import QueryRewriter

    rewriter = QueryRewriter(llm)
    rewritten = await rewriter.rewrite("什么是RAG？")

    print(f"  原始查询: '什么是RAG？'")
    print(f"  改写结果 ({len(rewritten)} 条):")
    for i, rw in enumerate(rewritten, 1):
        print(f"    {i}. {rw}")
    print()
    print("  改写器说明:")
    print("    - HyDERewriter (t=0.3):   生成假设答案用于检索")
    print("    - KeywordRewriter (t=0):  提取 BM25 关键词（幂等）")
    print("    - SynonymRewriter (t=0.3): 生成同义变体（多样性）")
    print("    - 三个改写器并行执行，合并去重")

    # ── 7. QueryUnderstandingLayer 完整流程 ─────────────────────
    banner("7. QueryUnderstandingLayer 完整流程")

    from query import get_query_layer, reset_query_layer

    layer = get_query_layer(llm)
    ctx = await layer.process("什么是带薪年休假？")

    print(f"  输入: '什么是带薪年休假？'")
    print(f"  意图: {ctx.intent.value if ctx.intent else 'N/A'}")
    print(f"  改写查询: {ctx.rewritten_queries}")
    print(f"  需要澄清: {ctx.needs_clarification}")
    print(f"  Pipeline 流程: 别名映射 → 意图分类 → 上下文融合 → 查询改写")

    # ── 8. 模糊问题短路 ─────────────────────────────────────────
    banner("8. 模糊问题短路")

    ctx2 = await layer.process("帮帮我")
    print(f"  输入: '帮帮我'")
    print(f"  需要澄清: {ctx2.needs_clarification}")
    if ctx2.clarification_question:
        print(f"  澄清问题: {ctx2.clarification_question!r}")
    print(f"  说明: 高模糊问题直接返回澄清请求，不进入后续检索/生成")

    # ── 清理 ────────────────────────────────────────────────────
    reset_query_layer()

    # ── 总结 ───────────────────────────────────────────────────
    print(f"\n  🔢 MockLLM 共接收 {len(llm.calls)} 次调用")
    banner("✅ 查询理解模块演示完成")
    print()
    print("  温度约定:")
    print("    IntentClassifier  (t=0)   确定性意图分类")
    print("    ContextFuser       (t=0)   确定性指代消解")
    print("    KeywordRewriter    (t=0)   幂等关键词提取")
    print("    HyDERewriter       (t=0.3) 受控假设答案")
    print("    SynonymRewriter    (t=0.3) 多样性同义变体")


if __name__ == "__main__":
    asyncio.run(main())
