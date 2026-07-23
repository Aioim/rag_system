"""
02_secrets_storage.py — 安全管理：内存加密存储与 .env 加载

演示内容：
  1. SecretsManager 内存加密存储（set/get）
  2. 环境变量 ENC[...] 解密流程
  3. SecureEnvLoader 自动加载机制
  4. process_env_file 批量加密 .env

运行方式：
  cd rag0709
  python examples/02_security/02_secrets_storage.py
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
    # ── 1. SecretsManager 内存加密存储 ──────────────────────────
    banner("1. SecretsManager 内存加密存储")

    from security import secrets, get_secret, set_secret

    secrets.set_secret("LLM_API_KEY", "sk-deepseek-v4-pro-xxx")
    secrets.set_secret("HUGGINGFACE_TOKEN", "hf_secret_token_yyy")
    secrets.set_secret("DB_PASSWORD", "secure_db_pass_123")

    print(f"  LLM_API_KEY        = {get_secret('LLM_API_KEY')!r}")
    print(f"  HUGGINGFACE_TOKEN  = {get_secret('HUGGINGFACE_TOKEN')!r}")
    print(f"  DB_PASSWORD        = {get_secret('DB_PASSWORD')!r}")
    print(f"  不存在的 key       = {get_secret('NOT_EXIST')!r}")

    key_count = len(secrets.list_secrets())
    print(f"\n  已存储密钥数量: {key_count}")

    # 空值
    secrets.set_secret("EMPTY_KEY", "")
    print(f"  空值密钥: {get_secret('EMPTY_KEY')!r}")

    # ── 2. 环境变量 ENC[...] 解密 ───────────────────────────────
    banner("2. 环境变量 ENC[...] 解密流程")

    from security import fetch_and_decrypt_env_var, process_env_file
    from security import encrypt_value

    try:
        llm_key = fetch_and_decrypt_env_var("LLM_API_KEY")
        print(f"  LLM_API_KEY 解密后: {llm_key[:20]}...")
    except Exception:
        print(f"  LLM_API_KEY: 解密失败（.secret_key 可能已更换），需重新加密")

    # process_env_file 演示 — 批量加密 .env 中的敏感字段
    import tempfile
    from pathlib import Path

    test_enc = encrypt_value("demo-token-for-test")
    sample_content = f"""
# 明文值
PUBLIC_KEY=not-secret-value
# 加密值
SECRET_TOKEN=ENC[{test_enc}]
""".strip()

    tmp_env = Path(tempfile.mkdtemp()) / ".env.demo"
    tmp_env.write_text(sample_content)
    try:
        process_env_file(str(tmp_env))
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

    # ── 3. SecureEnvLoader ──────────────────────────────────────
    banner("3. SecureEnvLoader 自动加载")

    from security import SecureEnvLoader

    loader = SecureEnvLoader()
    print(f"  loader 实例: {loader!r}")
    print()
    print("  ConfigManager 初始化时自动调用:")
    print("    1. dotenv.load_dotenv() — 加载明文 .env")
    print("    2. SecureEnvLoader.load(override=True) — 解密 ENC[...] 覆盖")
    print("  应用代码通常无需手动调用 load_secure_dotenv()")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 密钥存储演示完成")
    print()
    print("  CLI 工具:")
    print("    python -m src.security <KEY>              # 交互式加密")
    print("    python -m src.security --encrypt-file .env # 批量加密")
    print("    python -m src.security --decrypt <cipher> # 解密验证")
    print()
    print("  最佳实践:")
    print("    1. 禁止在 YAML / 代码中硬编码密钥")
    print("    2. 敏感值统一写入 .env，加密字段用 ENC[...] 包裹")
    print("    3. .env 文件已加入 .gitignore")


if __name__ == "__main__":
    main()
