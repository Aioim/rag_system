"""
demo_core.py — RAG Core Pipeline 模块演示

演示内容：
  1. RAGPipeline 全链路编排
  2. 单轮问答流程
  3. 多轮对话（含会话上下文）
  4. 模糊问题短路
  5. 各层异常独立降级
  6. Fallback 触发场景

运行方式：
  cd rag0709
  python examples/10_core/demo_core.py
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
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    # ── 1. 初始化 ───────────────────────────────────────────────
    banner("1. 初始化 RAGPipeline")

    class MockLLM:
        """完整 Mock LLM — 模拟各层所需的 LLM 响应"""

        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        async def ainvoke(self, prompt: str, **kwargs):
            self.calls.append((prompt[:200], kwargs))

            # 意图分类
            if "意图分类" in prompt or "intent" in prompt.lower():
                if "什么是" in prompt:
                    return SimpleNamespace(content='{"intent": "concept", "confidence": 0.92}')
                if "怎么" in prompt or "如何" in prompt or "申请" in prompt or "材料" in prompt:
                    return SimpleNamespace(content='{"intent": "procedure", "confidence": 0.90}')
                if "区别" in prompt or "比较" in prompt:
                    return SimpleNamespace(content='{"intent": "compare", "confidence": 0.85}')
                if "电话" in prompt or "地址" in prompt or "多少" in prompt:
                    return SimpleNamespace(content='{"intent": "lookup", "confidence": 0.95}')
                if "帮帮我" in prompt:
                    return SimpleNamespace(content='{"intent": "concept", "confidence": 0.30}')
                return SimpleNamespace(content='{"intent": "concept", "confidence": 0.88}')

            # 清晰度
            if "清晰度" in prompt or "模糊" in prompt:
                if "帮帮我" in prompt:
                    return SimpleNamespace(content='{"needs_clarification": true, "question": "您想了解哪方面内容？"}')
                return SimpleNamespace(content='{"needs_clarification": false, "question": null}')

            # 上下文融合
            if "上下文" in prompt or "补全" in prompt:
                return SimpleNamespace(content='{"resolved_query": "申请带薪年休假需要什么材料？"}')

            # HyDE
            if "假设答案" in prompt:
                return SimpleNamespace(content="假设的年假申请流程相关段落...")

            # 同义变体
            if "同义" in prompt:
                return SimpleNamespace(content='["带薪年休假怎么申请", "年假申请流程", "申请年假步骤"]')

            # 关键词
            if "关键词" in prompt or "BM25" in prompt:
                return SimpleNamespace(content='["年假", "带薪年休假", "申请", "材料"]')

            # 生成回答 (默认/Pro 模型)
            return SimpleNamespace(
                content="""带薪年休假是员工依法享有的福利假期。

根据《员工手册》规定，工作满1年不满10年的员工每年享受5天带薪年假。

申请流程：登录OA系统 → 填写年假申请单 → 部门主管审批 → HR确认。

注意事项：年假可累积至次年3月31日。"""
            )

    from session.store import SessionStore
    from session.manager import SessionManager
    from core import get_rag_pipeline, reset_rag_pipeline

    llm = MockLLM()

    # 临时会话数据库
    tmp_dir = Path(tempfile.mkdtemp())
    store = SessionStore(db_path=tmp_dir / "demo_core.db")
    session_manager = SessionManager(store=store)

    pipeline = get_rag_pipeline(llm, session_manager)
    print("  ✅ RAGPipeline 已初始化")
    print(f"  注入: LLM + SessionManager")
    print(f"  包含子层: QueryUnderstandingLayer + RetrievalLayer + GenerationLayer + FallbackHandler")

    # ── 2. 单轮问答 ─────────────────────────────────────────────
    banner("2. 单轮问答 — 基础流程")

    ctx = await pipeline.run("什么是带薪年休假？")

    print(f"  查询: '什么是带薪年休假？'")
    print(f"  意图: {ctx.intent.value if ctx.intent else 'N/A'}")
    print(f"  回答: {ctx.answer[:120]}...")
    print(f"  置信度: {ctx.confidence}")
    print(f"  触发兜底: {ctx.is_fallback}")
    print(f"  兜底级别: {ctx.fallback_level.value}")
    print(f"  Pipeline 流程: 查询理解 → 检索 → 兜底检查 → 生成 → 会话记录")

    # ── 3. 多轮对话 ─────────────────────────────────────────────
    banner("3. 多轮对话 — 含会话上下文")

    # 先创建会话
    session = session_manager.get_or_create()
    session_manager.add_message(session.session_id, "user", "带薪年休假怎么申请？")
    session_manager.add_message(
        session.session_id, "assistant",
        "申请带薪年休假需要登录OA系统，填写申请单后提交主管审批。"
    )

    # 追问
    ctx2 = await pipeline.run("需要什么材料？", session_id=session.session_id)

    print(f"  第1轮: '带薪年休假怎么申请？'")
    print(f"  第2轮: '需要什么材料？' (不完整的追问)")
    print(f"  回答: {ctx2.answer[:120]}...")
    print(f"  说明: 自动补全为独立查询后再检索生成")

    # ── 4. 模糊问题短路 ─────────────────────────────────────────
    banner("4. 模糊问题短路")

    ctx3 = await pipeline.run("帮帮我")

    print(f"  查询: '帮帮我'")
    print(f"  需要澄清: {ctx3.needs_clarification}")
    print(f"  澄清问题: {ctx3.clarification_question!r}")
    print(f"  说明: 高模糊问题直接返回澄清请求，不调用检索/生成层")

    # ── 5. Pipeline 异常降级 ────────────────────────────────────
    banner("5. Pipeline 异常独立降级")

    print("  各层异常处理策略:")
    print("    - 查询理解层失败 → 使用原始 query，意图默认为 concept")
    print("    - 检索层失败     → candidates/reranked 为空，触发兜底")
    print("    - 兜底层失败     → 降级到下一级兜底策略")
    print("    - 生成层失败     → 返回错误提示消息")
    print("    - 会话记录失败   → 仅记录日志，不影响回答返回")
    print()
    print("  ✅ 每层失败时记录日志并继续，不中断 Pipeline")

    # ── 6. Fallback 触发场景 ────────────────────────────────────
    banner("6. Fallback 触发场景")

    print("  三级兜底链路:")
    print("    1️⃣  NEED_MORE    → 补充检索（放宽 top_k）→ PARTIAL")
    print("    2️⃣  INSUFFICIENT → 联网搜索（DuckDuckGo）→ WEB_SEARCH")
    print("    3️⃣  搜索失败     → 诚实告知 → NO_ANSWER")
    print()
    print("  兜底配置:")
    from config import settings
    print(f"    max_retrieval_rounds:  {settings.fallback.max_retrieval_rounds}")
    print(f"    web_search.enabled:    {settings.web_search.enabled}")
    print(f"    web_search.provider:   {settings.web_search.provider}")
    print(f"    web_search.timeout:    {settings.web_search.timeout_seconds}s")
    print(f"    no_answer_message:     {settings.fallback.no_answer_message[:40]}...")

    # ── 7. Pipeline 元数据 ──────────────────────────────────────
    banner("7. Pipeline 运行统计")

    print(f"  MockLLM 总调用次数: {len(llm.calls)}")
    llm_call_types = {}
    for call_prompt, _ in llm.calls:
        # 粗略分类
        if "意图" in call_prompt:
            key = "意图分类/清晰度"
        elif "上下文" in call_prompt or "补全" in call_prompt:
            key = "上下文融合"
        elif "假设" in call_prompt:
            key = "HyDE改写"
        elif "同义" in call_prompt:
            key = "同义变体"
        elif "关键词" in call_prompt:
            key = "关键词提取"
        else:
            key = "生成回答"
        llm_call_types[key] = llm_call_types.get(key, 0) + 1

    for key, count in sorted(llm_call_types.items()):
        print(f"    {key}: {count} 次")

    # ── 清理 ────────────────────────────────────────────────────
    reset_rag_pipeline()
    store.close()

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ Core Pipeline 模块演示完成")
    print()
    print("  编程接口:")
    print("    from core import get_rag_pipeline")
    print("    pipeline = get_rag_pipeline(llm, session_manager)")
    print("    ctx = await pipeline.run('查询', session_id='可选')")
    print()
    print("  返回值 PipelineContext 包含:")
    print("    answer, sources, confidence, is_fallback, fallback_level,")
    print("    intent, rewritten_queries, reranked, assembled_prompt")


if __name__ == "__main__":
    asyncio.run(main())
