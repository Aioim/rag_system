"""
03_data_masking.py — 日志系统：敏感数据脱敏

演示内容：
  1. mask_sensitive_data — 常见敏感信息脱敏
  2. MaskingEngine — 脱敏引擎
  3. 自定义脱敏规则示例

运行方式：
  cd rag0709
  python examples/03_logger/03_data_masking.py
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
    # ── 1. 敏感数据脱敏 ─────────────────────────────────────────
    banner("1. 敏感数据脱敏 (mask_sensitive_data)")

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

    # ── 2. MaskingEngine 脱敏引擎 ───────────────────────────────
    banner("2. MaskingEngine — 脱敏引擎")

    from logger import MaskingEngine

    engine = MaskingEngine()
    print(f"  MaskingEngine 实例: {engine!r}")
    print(f"  使用 mask_sensitive_data() 即可自动脱敏常见敏感信息")

    # 更多脱敏场景
    extra_tests = [
        "Bank card: 6222021234567890123",
        "IP: 192.168.1.100",
    ]
    for text in extra_tests:
        masked = mask_sensitive_data(text)
        result = f"  {text} → {masked}" if masked != text else f"  {text} → (不变)"
        print(result)

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 数据脱敏演示完成")
    print()
    print("  脱敏规则覆盖:")
    print("    - API Key / Token / 密码")
    print("    - 手机号 / 身份证号 / 银行卡号")
    print("    - 邮箱地址 / IP 地址")
    print()
    print("  SensitiveDataFilter — 自动过滤日志中的敏感信息")
    print("  SecurityAuditFilter — 安全审计日志专用过滤器")


if __name__ == "__main__":
    main()
