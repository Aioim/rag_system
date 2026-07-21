"""
demo_fallback.py — 兜底处理模块演示

演示内容：
  1. WebSearcher — 联网搜索
  2. SupplementaryRetriever — 补充检索
  3. FallbackHandler — 三级兜底编排
  4. NEED_MORE → PARTIAL 流程
  5. INSUFFICIENT → WEB_SEARCH / NO_ANSWER 流程

运行方式：
  cd rag0709
  python examples/12_fallback/demo_fallback.py

注意：
  - 联网搜索需要网络连接
  - 补充检索需要 FAISS 索引（先运行 ingestion）
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
    from config import settings

    # ── 1. WebSearcher — 联网搜索 ───────────────────────────────
    banner("1. WebSearcher — 联网搜索")

    from fallback.web_search import WebSearcher

    searcher = WebSearcher()

    print(f"  搜索提供商: {settings.web_search.provider}")
    print(f"  超时:       {settings.web_search.timeout_seconds}s")
    print(f"  启用状态:   {'✅ 已启用' if settings.web_search.enabled else '⚠️ 已禁用'}")

    if settings.web_search.enabled:
        print("\n  测试搜索: 'Python RAG框架'")
        try:
            result = await searcher.search("Python RAG 框架")
            if result:
                print(f"  ✅ 搜索成功，结果长度: {len(result)} 字符")
                print(f"  结果预览:\n---\n{result[:300]}...\n---")
            else:
                print("  ⚠️ 搜索返回空结果")
        except Exception as e:
            print(f"  ⚠️ 搜索失败: {e}")
            print("  (网络问题或 DuckDuckGo 不可用是正常的)")
    else:
        print("  跳过联网搜索测试（web_search.enabled = False）")

    # ── 2. SupplementaryRetriever ───────────────────────────────
    banner("2. SupplementaryRetriever — 补充检索")

    from fallback.supplementary import SupplementaryRetriever
    from models.context import PipelineContext
    from models.enums import RetrievalEval

    supp = SupplementaryRetriever()
    print(f"  补充检索器已创建")

    # 检查索引是否存在
    from config import PROJECT_ROOT
    faiss_dir = PROJECT_ROOT / settings.faiss.index_dir
    if list(faiss_dir.glob("**/index.faiss")):
        from models.chunk import Chunk

        # 模拟 NEED_MORE 场景
        ctx = PipelineContext(query="年假申请流程")
        ctx.retrieval_eval = RetrievalEval.NEED_MORE
        ctx.candidates = [
            Chunk(chunk_id="c1", doc_id="d1", text="员工手册包含考勤和休假章节",
                  chunk_index=0, rerank_score=0.45),
        ]

        try:
            from retrieval import get_retrieval_layer
            layer = get_retrieval_layer()
            ctx = await supp.retrieve(ctx, layer)
            print(f"  补充检索前: {len(ctx.candidates)} 条候选")
            print(f"  补充检索后: {len(ctx.candidates)} 条候选")
            print(f"  说明: 放宽 top_k 重新检索，合并结果")
        except Exception as e:
            print(f"  ⚠️ 补充检索失败: {e}")
            print(f"  (需要先下载 Embedding 模型: models.download('embedding'))")
    else:
        print("  ⚠️ 未找到 FAISS 索引，跳过补充检索演示")
        print("  请先运行: python examples/11_ingestion/demo_ingestion.py")

    # ── 3. FallbackHandler 架构 ─────────────────────────────────
    banner("3. FallbackHandler — 三级兜底编排")

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

    # ── 4. NEED_MORE → PARTIAL 流程（概念演示） ─────────────────
    banner("4. NEED_MORE → PARTIAL 流程演示")

    print("  配置:")
    print(f"    max_retrieval_rounds: {settings.fallback.max_retrieval_rounds}")
    print()
    print("  流程:")
    print("    1. RetrievalLayer.retrieve() → NEED_MORE")
    print("    2. FallbackHandler.handle(NEED_MORE)")
    print("    3. SupplementaryRetriever.retrieve() — 放宽 top_k × 2")
    print("    4. 合并新结果 → Rerank")
    print("    5. 若仍 NEED_MORE → 最多再试 {max_retrieval_rounds} 轮".format(
        max_retrieval_rounds=settings.fallback.max_retrieval_rounds))
    print("    6. 标记 fallback_level = PARTIAL，进入生成")

    # ── 5. INSUFFICIENT → WEB_SEARCH / NO_ANSWER ────────────────
    banner("5. INSUFFICIENT → WEB_SEARCH / NO_ANSWER 流程演示")

    print("  流程:")
    print("    1. RetrievalLayer.retrieve() → INSUFFICIENT")
    print("    2. FallbackHandler.handle(INSUFFICIENT)")
    print("    3. WebSearcher.search(query)")
    print("    4. 搜索成功 → answer = 搜索结果 + 标注来源")
    print("    5. 搜索失败 → answer = no_answer_message")
    print()
    print("  搜索结果处理:")
    print("    - 截取 top_results 条（web_search.max_results）")
    print("    - 拼接为上下文 → 调用 LLM 生成回答")
    print("    - 添加来源标注 [Web Search]")

    # ── 6. 兜底配置 ─────────────────────────────────────────────
    banner("6. 完整兜底配置")

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
    banner("✅ 兜底处理模块演示完成")
    print()
    print("  编程接口:")
    print("    from fallback import get_fallback_handler, WebSearcher, SupplementaryRetriever")
    print("    handler = get_fallback_handler()")
    print("    ctx = await handler.handle(ctx, retrieval_layer)")
    print()
    print("  FallbackLevel 枚举:")
    print("    NONE       — 未触发兜底")
    print("    PARTIAL    — 资料不足但尝试生成")
    print("    WEB_SEARCH — 触发联网搜索")
    print("    NO_ANSWER  — 诚实告知无法回答")


if __name__ == "__main__":
    asyncio.run(main())
