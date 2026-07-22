"""
01_secret_crypto.py — 安全管理：防泄露容器与 Fernet 加解密

演示内容：
  1. SecretStr 防泄露字符串容器
  2. Fernet 加密 / 解密
  3. 密钥文件生成
  4. 边界情况（空值、特殊字符）

运行方式：
  cd rag0709
  python examples/02_security/01_secret_crypto.py
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
    # ── 1. SecretStr 防泄露字符串 ───────────────────────────────
    banner("1. SecretStr — 防泄露字符串容器")

    from security import SecretStr

    token = SecretStr("sk-abc123def456")
    print(f"  token = {token!r}")                 # <SecretStr: ***>
    print(f"  str(token) = {str(token)}")          # ***
    print(f"  token.get() = {token.get()!r}")      # 显式获取
    print()
    print("  ✓ SecretStr 无法被意外打印、序列化或日志记录")

    same = SecretStr("sk-abc123def456")
    print(f"  token == same: {token == same}")    # True (值相等)

    # ── 2. Fernet 加密 / 解密 ───────────────────────────────────
    banner("2. Fernet 加密 / 解密")

    from security import encrypt_value, decrypt_value, generate_key_file

    plaintext = "my-secret-api-key-12345"
    encrypted = encrypt_value(plaintext)
    print(f"  原始值:   {plaintext!r}")
    print(f"  加密后:   {encrypted[:40]}...")

    decrypted = decrypt_value(encrypted)
    print(f"  解密后:   {decrypted!r}")
    print(f"  是否一致: {plaintext == decrypted}")

    # ── 3. 密钥文件生成 ─────────────────────────────────────────
    banner("3. 密钥文件生成")

    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.mkdtemp())
    key_path = tmp_dir / ".secret_key"

    generate_key_file(key_path)
    print(f"  密钥文件:     {key_path}")
    print(f"  文件存在:     {key_path.exists()}")
    print(f"  文件大小:     {key_path.stat().st_size} bytes")
    print()
    print("  生产环境: python -m security.env_encrypt generate-key")

    # ── 4. 边界情况 ─────────────────────────────────────────────
    banner("4. 边界情况")

    # 特殊字符
    special = "sk-!@#$%^&*()_+-=[]{}|;':\",./<>?"
    encrypted_special = encrypt_value(special)
    decrypted_special = decrypt_value(encrypted_special)
    print(f"  特殊字符加密后解密: {special == decrypted_special}")

    # 非 ENC[...] 格式
    from security import fetch_and_decrypt_env_var
    print(f"\n  fetch_and_decrypt_env_var:")
    print(f"    只接受 ENC[...] 格式的加密值")
    print(f"    普通环境变量请用 os.environ.get() 直接读取")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 加解密演示完成")
    print()
    print("  下一步: 02_secrets_storage.py — SecretsManager 存储与 .env 解密")


if __name__ == "__main__":
    main()
