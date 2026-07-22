"""
01_web_search.py — 兜底处理：联网搜索

演示内容：
  1. WebSearcher — 联网搜索（真实 DuckDuckGo 搜索）
  2. 搜索提供商配置
  3. 超时与错误处理

运行方式：
  cd rag0709
  python examples/12_fallback/01_web_search.py

注意：联网搜索需要网络连接
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
    # ── 1. WebSearcher 配置 ─────────────────────────────────────
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
        print("  启用方式: 修改 config/{env}.yaml → web_search.enabled = true")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 联网搜索演示完成")
    print()
    print("  下一步: 02_fallback_flow.py — FallbackHandler 三级兜底流程")


if __name__ == "__main__":
    asyncio.run(main())
