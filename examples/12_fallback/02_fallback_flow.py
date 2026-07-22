"""
02_fallback_flow.py — 兜底处理：三级兜底编排

演示内容：
  1. SupplementaryRetriever — 补充检索
  2. FallbackHandler — 三级兜底架构
  3. NEED_MORE → PARTIAL 流程
  4. INSUFFICIENT → WEB_SEARCH / NO_ANSWER 流程
  5. 兜底配置详解

运行方式：
  cd rag0709
  python examples/12_fallback/02_fallback_flow.py
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
    # ── 1. FallbackHandler 架构 ─────────────────────────────────
    banner("1. FallbackHandler — 三级兜底编排")

    from fallback import get_fallback_handler, reset_fallback_handler

    handler = get_fallback_handler()
    print(f"  FallbackHandler 已创建 (单例)")

    print("\n  三级兜底链路:")
    print("  ┌─ Level 0: NONE")
    print("  │  检索结果 SUFFICIENT，不触发兜底")
    print("  │")
    print("  ├─ Level 1: PARTIAL")
    print("  │  NEED_MORE → 补充检索（放宽 top_k，max 2 轮）")
    print("  │  合并结果后尝试生成，标记 partial")
    print("  │")
    print("  ├─ Level 2: WEB_SEARCH")
    print("  │  INSUFFICIENT → 联网搜索（DuckDuckGo）")
    print("  │  搜索成功 → 用搜索结果作为上下文生成")
    print("  │")
    print("  └─ Level 3: NO_ANSWER")
    print("      联网搜索失败 → 诚实告知无法回答")
    print(f"      消息: {settings.fallback.no_answer_message[:50]}...")

    # ── 2. NEED_MORE → PARTIAL 流程 ─────────────────────────────
    banner("2. NEED_MORE → PARTIAL 流程")

    print("  配置:")
    print(f"    max_retrieval_rounds: {settings.fallback.max_retrieval_rounds}")
    print()
    print("  流程:")
    print("    1. RetrievalLayer.retrieve() → NEED_MORE")
    print("    2. FallbackHandler.handle(NEED_MORE)")
    print("    3. SupplementaryRetriever.retrieve() — 放宽 top_k × 2")
    print("    4. 合并新结果 → Rerank")
    print("    5. 若仍 NEED_MORE → 最多再试 N 轮")
    print("    6. 标记 fallback_level = PARTIAL，进入生成")

    # ── 3. INSUFFICIENT → WEB_SEARCH / NO_ANSWER ────────────────
    banner("3. INSUFFICIENT → WEB_SEARCH / NO_ANSWER 流程")

    print("  流程:")
    print("    1. RetrievalLayer.retrieve() → INSUFFICIENT")
    print("    2. FallbackHandler.handle(INSUFFICIENT)")
    print("    3. WebSearcher.search(query)")
    print("    4. 搜索成功 → answer = 搜索结果 + 标注来源")
    print("    5. 搜索失败 → answer = no_answer_message")
    print()
    print("  搜索结果处理:")
    print("    - 截断至 top_results 条")
    print("    - 拼接为上下文 → 调用 LLM 生成回答")
    print("    - 添加来源标注 [Web Search]")

    # ── 4. 完整兜底配置 ─────────────────────────────────────────
    banner("4. 完整兜底配置")

    print(f"  补充检索:")
    print(f"    max_retrieval_rounds: {settings.fallback.max_retrieval_rounds}")
    print(f"    (放宽后的 top_k = retrieval.top_k × 2)")
    print()
    print(f"  联网搜索:")
    print(f"    enabled:            {settings.web_search.enabled}")
    print(f"    provider:           {settings.web_search.provider}")
    print(f"    timeout_seconds:    {settings.web_search.timeout_seconds}")
    print()
    print(f"  诚实告知:")
    print(f"    no_answer_message:  {settings.fallback.no_answer_message}")

    # ── 清理 ────────────────────────────────────────────────────
    reset_fallback_handler()

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 兜底处理演示完成")
    print()
    print("  FallbackLevel 枚举:")
    print("    NONE       — 未触发兜底")
    print("    PARTIAL    — 资料不足但尝试生成")
    print("    WEB_SEARCH — 触发联网搜索")
    print("    NO_ANSWER  — 诚实告知无法回答")


if __name__ == "__main__":
    main()
