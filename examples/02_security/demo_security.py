"""
demo_security.py — 安全管理模块演示

演示内容：
  1. SecretStr 防泄露字符串容器
  2. SecretsManager 内存加密存储
  3. Fernet 加密 / 解密工具函数
  4. .env 文件安全加载（含 ENC[...] 解密）
  5. 密钥文件生成
  6. 读取已加密的环境变量

运行方式：
  cd rag0709
  python examples/02_security/demo_security.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 先触发配置初始化，打破 logger → settings → security 循环导入
from config import settings  # noqa: E402, F401
_ = settings.env  # 强制触发 initialize()，在导入 security 前完成 _config 设置  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. SecretStr 防泄露字符串 ───────────────────────────────
    banner("1. SecretStr — 防泄露字符串容器")

    from security import SecretStr

    token = SecretStr("sk-abc123def456")
    print(f"  token = {token!r}")                 # 显示为 <SecretStr: ***>
    print(f"  str(token) = {str(token)}")          # 显示为 ***
    print(f"  token.get() = {token.get()!r}")      # 显式获取原始值
    print()
    print("  ✓ SecretStr 无法被意外打印、序列化或日志记录")

    # 比较两个 SecretStr
    same = SecretStr("sk-abc123def456")
    print(f"  token == same: {token == same}")    # True
    # 注意: SecretStr 故意设为不可哈希 (安全性保护)

    # ── 2. Fernet 加密 / 解密 ───────────────────────────────────
    banner("2. Fernet 加密 / 解密")

    from security import encrypt_value, decrypt_value, generate_key_file

    # 使用默认密钥文件加密
    plaintext = "my-secret-api-key-12345"
    encrypted = encrypt_value(plaintext)
    print(f"  原始值: {plaintext!r}")
    print(f"  加密后: {encrypted[:40]}...")

    decrypted = decrypt_value(encrypted)
    print(f"  解密后: {decrypted!r}")
    print(f"  是否一致: {plaintext == decrypted}")

    # ── 3. 密钥文件生成 ─────────────────────────────────────────
    banner("3. 密钥文件生成")

    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.mkdtemp())
    key_path = tmp_dir / ".secret_key"

    generate_key_file(key_path)
    print(f"  密钥文件已生成: {key_path}")
    print(f"  文件是否存在: {key_path.exists()}")
    print(f"  文件大小: {key_path.stat().st_size} bytes")
    print()
    print("  日常使用: 首次部署时运行 python -m security.env_encrypt generate-key")

    # ── 4. SecretsManager 内存加密存储 ──────────────────────────
    banner("4. SecretsManager 内存加密存储")

    from security import secrets, get_secret, set_secret

    # 存储多个密钥
    secrets.set_secret("LLM_API_KEY", "sk-deepseek-v4-pro-xxx")
    secrets.set_secret("HUGGINGFACE_TOKEN", "hf_secret_token_yyy")
    secrets.set_secret("DB_PASSWORD", "secure_db_pass_123")

    # 读取
    print(f"  LLM_API_KEY      = {get_secret('LLM_API_KEY')!r}")
    print(f"  HUGGINGFACE_TOKEN = {get_secret('HUGGINGFACE_TOKEN')!r}")
    print(f"  DB_PASSWORD       = {get_secret('DB_PASSWORD')!r}")

    # 不存在的 key
    print(f"  不存在的 key      = {get_secret('NOT_EXIST')!r}")

    # 查看已存储 key 数量
    key_count = len(secrets._store) if hasattr(secrets, "_store") else "N/A"
    print(f"\n  已存储密钥数量: {key_count}")

    # ── 5. 环境变量加密/解密流程 ───────────────────────────────
    banner("5. 环境变量 ENC[...] 解密流程")

    from security import fetch_and_decrypt_env_var, process_env_file

    # 尝试解密当前环境中的加密变量（可能因密钥不匹配失败）
    try:
        llm_key = fetch_and_decrypt_env_var("LLM_API_KEY")
        print(f"  LLM_API_KEY 解密后: {llm_key[:20]}...")
    except Exception:
        print(f"  LLM_API_KEY: 解密失败（.secret_key 可能已更换），需重新加密")

    # 演示 process_env_file — 批量加密 .env 文件中的敏感字段
    import tempfile
    test_enc = encrypt_value("demo-token-for-test")
    sample_content = f"""
# 明文值
PUBLIC_KEY=not-secret-value
# 加密值
SECRET_TOKEN=ENC[{test_enc}]
""".strip()

    # process_env_file 接收文件路径，将敏感字段加密后写入新文件
    tmp_env = Path(tempfile.mkdtemp()) / ".env.demo"
    tmp_env.write_text(sample_content)
    try:
        process_env_file(str(tmp_env))
        # 输出文件: {input}.encrypted
        encrypted_file = Path(str(tmp_env) + ".encrypted")
        if encrypted_file.exists():
            print(f"  已生成加密文件: {encrypted_file}")
            print(f"  加密后内容预览:")
            for line in encrypted_file.read_text().strip().split("\n")[:3]:
                print(f"    {line[:60]}")
            encrypted_file.unlink(missing_ok=True)
    except SystemExit:
        print("  (process_env_file 演示跳过)")
    finally:
        tmp_env.unlink(missing_ok=True)

    # ── 6. SecureEnvLoader ──────────────────────────────────────
    banner("6. SecureEnvLoader")

    from security import SecureEnvLoader

    loader = SecureEnvLoader()
    print(f"  loader 实例: {loader!r}")
    print()
    print("  ConfigManager 初始化时会自动调用:")
    print("    1. dotenv.load_dotenv() — 加载明文 .env")
    print("    2. SecureEnvLoader.load(override=True) — 解密 ENC[...] 覆盖")
    print("  应用代码通常无需手动调用 load_secure_dotenv()")

    # ── 7. 边界情况 ─────────────────────────────────────────────
    banner("7. 边界情况")

    # 空值
    secrets.set_secret("EMPTY_KEY", "")
    print(f"  空值密钥: {get_secret('EMPTY_KEY')!r}")

    # 特殊字符
    special = "sk-!@#$%^&*()_+-=[]{}|;':\",./<>?"
    encrypted_special = encrypt_value(special)
    decrypted_special = decrypt_value(encrypted_special)
    print(f"  特殊字符加密后解密: {special == decrypted_special}")

    # 非 ENC[...] 格式的值 — fetch_and_decrypt_env_var 会报错（预期的安全行为）
    print(f"  非 ENC 格式: fetch_and_decrypt_env_var 只接受 ENC[...] 格式的加密值")
    print(f"  普通环境变量请使用 os.environ.get() 直接读取")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 安全管理模块演示完成")
    print()
    print("  CLI 工具:")
    print("    python -m security.env_encrypt encrypt <value>   # 加密")
    print("    python -m security.env_encrypt decrypt <cipher>  # 解密")
    print("    python -m security.env_encrypt generate-key      # 生成密钥")
    print()
    print("  最佳实践:")
    print("    1. 禁止在 YAML / 代码中硬编码密钥")
    print("    2. 敏感值统一写入 .env，加密字段用 ENC[...] 包裹")
    print("    3. .env 文件已加入 .gitignore")


if __name__ == "__main__":
    main()
