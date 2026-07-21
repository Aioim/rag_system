"""
demo_logger.py — 日志系统模块演示

演示内容：
  1. 基础日志输出（logger / security_logger）
  2. 日志级别（DEBUG / INFO / WARNING / ERROR）
  3. @log_performance 性能监控装饰器
  4. @log_step 步骤追踪装饰器
  5. log_duration 上下文管理器
  6. 敏感数据脱敏 (mask_sensitive_data)
  7. MaskingEngine — 自定义脱敏规则
  8. SensitiveDataFilter — 日志过滤器

运行方式：
  cd rag0709
  python examples/03_logger/demo_logger.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402, F401
_ = settings.env  # 强制触发 initialize()，在导入 logger 前完成 _config 设置  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    from logger import logger, security_logger

    # ── 1. 基础日志输出 ─────────────────────────────────────────
    banner("1. 基础日志输出")

    logger.info("这是一条 INFO 日志 — 演示基础日志输出")
    logger.debug("这是一条 DEBUG 日志 — 调试详细信息")
    logger.warning("这是一条 WARNING 日志 — 警告信息")
    logger.error("这是一条 ERROR 日志 — 错误信息")

    security_logger.info("安全审计日志: 用户登录成功")
    print("  ✓ 已输出 5 条日志（查看控制台和 logs/ 目录）")

    # ── 2. 日志级别过滤 ─────────────────────────────────────────
    banner("2. 日志级别过滤")

    import logging

    # 获取 logger 实例并检查级别
    print(f"  logger 级别: {logging.getLevelName(logger.level)}")
    print(f"  security_logger 级别: {logging.getLevelName(security_logger.level)}")
    print(f"  当前配置级别: INFO（DEBUG 不可见）")
    print(f"  可通过 env.yaml → log.log_level 调整")

    # ── 3. @log_performance 性能监控 ─────────────────────────────
    banner("3. @log_performance 性能监控装饰器")

    from logger import log_performance

    @log_performance(threshold_ms=50)
    def slow_operation(n: int) -> int:
        """模拟耗时操作"""
        time.sleep(0.1)
        return n * 2

    @log_performance(threshold_ms=50)
    def fast_operation(n: int) -> int:
        """快速操作（不触发警告）"""
        return n * 2

    result_slow = slow_operation(5)
    result_fast = fast_operation(10)
    print(f"  slow_operation(5) = {result_slow}  → 超过 50ms 阈值会记录 WARNING")
    print(f"  fast_operation(10) = {result_fast} → 未超过阈值不记录")

    # ── 4. @log_step 步骤追踪装饰器 ─────────────────────────────
    banner("4. @log_step 步骤追踪装饰器")

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

    # ── 5. log_duration 上下文管理器 ────────────────────────────
    banner("5. log_duration 上下文管理器")

    from logger import log_duration

    with log_duration("批量 Embedding"):
        time.sleep(0.15)
        print("  ... 处理中 ...")

    print("  ✓ 自动记录进入/退出和耗时")

    # ── 6. 敏感数据脱敏 ─────────────────────────────────────────
    banner("6. 敏感数据脱敏 (mask_sensitive_data)")

    from logger import mask_sensitive_data

    test_texts = [
        "API Key: sk-deepseek-v4-pro-abc123xyz",
        "密码: MyP@ssw0rd!2024",
        "Token: Bearer eyJhbGciOiJIUzI1NiJ9.xxx",
        "手机号: 13812345678",
        "身份证: 110101199001011234",
        "邮箱: admin@company.com",
    ]

    for text in test_texts:
        masked = mask_sensitive_data(text)
        if masked != text:
            print(f"  脱敏前: {text}")
            print(f"  脱敏后: {masked}")
            print()

    # ── 7. 自定义脱敏规则 ───────────────────────────────────────
    banner("7. MaskingEngine — 脱敏引擎")

    from logger import MaskingEngine

    engine = MaskingEngine()
    print(f"  MaskingEngine 实例: {engine!r}")
    print(f"  使用 mask_sensitive_data() 即可自动脱敏常见敏感信息")

    # 更多脱敏示例
    extra_tests = [
        "Bank card: 6222021234567890123",
        "IP: 192.168.1.100",
    ]
    for text in extra_tests:
        masked = mask_sensitive_data(text)
        result = f"  {text} → {masked}" if masked != text else f"  {text} → (不变)"
        print(result)

    # ── 8. 日志过滤器和异常记录 ─────────────────────────────────
    banner("8. 异常记录")

    from logger import log_exception, log_security_event

    try:
        raise ValueError("模拟一个业务异常")
    except ValueError:
        log_exception(context="处理请求时发生错误")

    log_security_event(
        action="未授权访问检测",
        user="admin",
        resource="/api/admin",
        status="blocked",
        details={"ip": "192.168.1.100", "reason": "invalid_token"}
    )
    print("  ✓ 异常和安全事件已记录")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 日志系统模块演示完成")
    print()
    print("  日志文件位置:")
    print(f"    - 应用日志: logs/rag_service.log")
    print(f"    - 安全日志: logs/security.log")
    print()
    print("  扩展阅读:")
    print("    - src/logger/README.md — 完整文档")
    print("    - SensitiveDataFilter — 自动过滤日志中的敏感信息")
    print("    - SecurityAuditFilter — 安全审计日志专用过滤器")


if __name__ == "__main__":
    main()
