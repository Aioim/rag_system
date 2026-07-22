"""
02_performance.py — 日志系统：性能监控与步骤追踪

演示内容：
  1. @log_performance 性能监控装饰器
  2. @log_step 步骤追踪装饰器
  3. log_duration 上下文管理器

运行方式：
  cd rag0709
  python examples/03_logger/02_performance.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. @log_performance 性能监控 ─────────────────────────────
    banner("1. @log_performance 性能监控装饰器")

    from logger import log_performance

    @log_performance(threshold_ms=50)
    def slow_operation(n: int) -> int:
        """模拟耗时操作 — 超过阈值会记录 WARNING"""
        time.sleep(0.1)
        return n * 2

    @log_performance(threshold_ms=50)
    def fast_operation(n: int) -> int:
        """快速操作 — 不超过阈值不记录"""
        return n * 2

    result_slow = slow_operation(5)
    result_fast = fast_operation(10)
    print(f"  slow_operation(5) = {result_slow}  → 超过 50ms 阈值，记录 WARNING")
    print(f"  fast_operation(10) = {result_fast} → 未超过阈值，不记录")

    # ── 2. @log_step 步骤追踪 ───────────────────────────────────
    banner("2. @log_step 步骤追踪装饰器")

    from logger import log_step

    @log_step("文档解析")
    def parse_document(path: str) -> dict:
        return {"path": path, "pages": 42, "format": "md"}

    @log_step("向量检索")
    def search_vectors(query: str) -> list:
        return [{"id": "c1", "score": 0.95}, {"id": "c2", "score": 0.88}]

    doc = parse_document("/data/report.pdf")
    results = search_vectors("什么是RAG")
    print(f"  parse_document → {doc}")
    print(f"  search_vectors → {results}")

    # ── 3. log_duration 上下文管理器 ────────────────────────────
    banner("3. log_duration 上下文管理器")

    from logger import log_duration

    with log_duration("批量 Embedding"):
        time.sleep(0.15)
        print("  ... 处理中 ...")

    print("  ✓ 自动记录进入/退出和耗时")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 性能监控演示完成")
    print()
    print("  下一步: 03_data_masking.py — 敏感数据脱敏")


if __name__ == "__main__":
    main()
