"""
01_basic_logging.py — 日志系统：基础输出与安全事件

演示内容：
  1. 基础日志输出（logger / security_logger）
  2. 日志级别过滤（DEBUG/INFO/WARNING/ERROR）
  3. 异常记录 (log_exception)
  4. 安全事件记录 (log_security_event)

运行方式：
  cd rag0709
  python examples/03_logger/01_basic_logging.py
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

    print(f"  logger 级别:          {logging.getLevelName(logger.level)}")
    print(f"  security_logger 级别: {logging.getLevelName(security_logger.level)}")
    print(f"  当前配置级别: INFO（DEBUG 不可见）")
    print(f"  可通过 config/{{env}}.yaml → log.log_level 调整")

    # ── 3. 异常记录 ─────────────────────────────────────────────
    banner("3. 异常记录 (log_exception)")

    from logger import log_exception

    try:
        raise ValueError("模拟一个业务异常")
    except ValueError:
        log_exception(context="处理请求时发生错误")

    print("  ✓ 异常已记录到日志文件")

    # ── 4. 安全事件记录 ─────────────────────────────────────────
    banner("4. 安全事件记录 (log_security_event)")

    from logger import log_security_event

    log_security_event(
        action="未授权访问检测",
        user="admin",
        resource="/api/admin",
        status="blocked",
        details={"ip": "192.168.1.100", "reason": "invalid_token"}
    )
    print("  ✓ 安全事件已记录")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 基础日志演示完成")
    print()
    print("  日志文件位置:")
    print(f"    - 应用日志: logs/rag_service.log")
    print(f"    - 安全日志: logs/security.log")
    print()
    print("  下一步: 02_performance.py — 性能监控与步骤追踪")


if __name__ == "__main__":
    main()
