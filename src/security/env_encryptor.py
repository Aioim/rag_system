"""
🔐 .env 敏感字段加密工具

提供命令行和编程接口，用于加密/解密 .env 文件中的敏感字段。

使用示例：
    # 加密单个值（交互式）
    $ python -m security.env_encrypt DB_PASSWORD

    # 批量加密整个 .env 文件
    $ python -m security.env_encrypt --encrypt-file .env

    # 解密验证
    $ python -m security.env_encrypt --decrypt ENC[gAAAA...]
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

from .secrets_manager import secrets as _global_secrets
from .secure_env_loader import SecureEnvLoader


def encrypt_value(value: str) -> str:
    """
    加密单个值并格式化为 ENC[...] 格式

    Args:
        value: 要加密的明文值

    Returns:
        str: ENC[base64_encoded_encrypted_value] 格式

    Example:
        #>>> encrypt_value("mysecretpassword")
        'ENC[gAAAAABkX9J3mZqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7]'
    """
    if not hasattr(_global_secrets, "encrypt_string"):
        raise RuntimeError("Fernet not initialized")
    return _global_secrets.encrypt_string(value)


def decrypt_value(encrypted_str: str) -> str:
    """
    解密 ENC[...] 格式的值

    Args:
        encrypted_str: ENC[base64_encoded_encrypted_value] 格式

    Returns:
        str: 解密后的明文值

    Raises:
        ValueError: 如果格式无效或解密失败

    Example:
        #>>> decrypt_value("ENC[gAAAAABkX9J3mZqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7XqV7]")
        'mysecretpassword'
    """
    if not encrypted_str.startswith("ENC[") or not encrypted_str.endswith("]"):
        raise ValueError("Invalid ENC format: must be ENC[...]")

    encrypted_b64 = encrypted_str[4:-1]
    if not hasattr(_global_secrets, "decrypt_string"):
        raise RuntimeError("Fernet not initialized")
    return _global_secrets.decrypt_string(encrypted_b64)

def fetch_and_decrypt_env_var(env_var: str) -> str:
    """
    解密 .env 文件中的加密字段

    Args:
        env_var: 环境变量名（例如：DB_PASSWORD）

    Returns:
        str: 解密后的明文值

    Raises:
        ValueError: 如果环境变量不存在或格式无效
        RuntimeError: 如果解密失败

    Example:
        #>>> fetch_and_decrypt_env_var("DB_PASSWORD")
        'mysecretpassword'
    """
    from dotenv import load_dotenv
    
    # 加载 .env 文件
    load_dotenv()
    
    # 获取环境变量值
    encrypted_value = os.getenv(env_var)
    if encrypted_value is None:
        raise ValueError(f"Environment variable '{env_var}' not found")
    
    # 检查是否为空值
    if not encrypted_value.strip():
        raise ValueError(f"Environment variable '{env_var}' is empty")
    
    try:
        # 解密值
        return decrypt_value(encrypted_value)
    except ValueError as e:
        # 重新抛出带有更详细信息的异常
        raise ValueError(f"Failed to decrypt '{env_var}': {e!s}") from e
    except Exception as e:
        raise RuntimeError(f"Error decrypting '{env_var}': {e!s}") from e


def process_env_file(input_path: str, output_path: str | None = None) -> None:
    """
    加密 .env 文件中的敏感字段

    自动识别包含敏感关键词的字段并加密

    Args:
        input_path: 输入 .env 文件路径
        output_path: 输出文件路径（默认为 {input}.encrypted）

    Example:
        #>>> process_env_file(".env.dev", ".env.dev.encrypted")
    """
    input_file = Path(input_path)
    output_file = Path(output_path) if output_path else input_file.with_suffix('.env.encrypted')

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # 识别敏感字段（基于命名约定）
    sensitive_keys = SecureEnvLoader.SECRET_KEY_PATTERNS
    encrypted_count = 0
    failed_count = 0

    with open(input_file, encoding='utf-8') as f_in:
        lines = f_in.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in lines:
            # 跳过空行/注释
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                f_out.write(line)
                continue

            # 解析 key=value
            if '=' not in line:
                f_out.write(line)
                continue

            parts = line.split('=', 1)
            if len(parts) != 2:
                f_out.write(line)
                continue

            key, value = parts
            key_stripped = key.strip().lower()
            value_stripped = value.strip()

            # 检查是否需要加密
            should_encrypt = (
                    any(k in key_stripped for k in sensitive_keys) and
                    not SecureEnvLoader.is_encrypted_value(value_stripped) and
                    not value_stripped.startswith('#')  # 非注释
            )

            if should_encrypt:
                # 移除引号（如果存在）
                clean_value = value_stripped.strip('"').strip("'")
                try:
                    encrypted = encrypt_value(clean_value)
                    f_out.write(f'{key}={encrypted}\n')
                    encrypted_count += 1
                    print(f"  🔒 Encrypted: {key.strip()}")
                except Exception as e:
                    print(f"  ❌ Failed to encrypt {key.strip()}: {e}", file=sys.stderr)
                    # 安全优先：加密失败时跳过该行，不将明文写入输出文件
                    failed_count += 1
            else:
                f_out.write(line)

    if failed_count > 0:
        print(f"\n⚠️  WARNING: {failed_count} field(s) could NOT be encrypted and were OMITTED from output.")
        print("   The output file is INCOMPLETE — resolve the errors above before deploying.")
    print(f"\n✅ Encrypted {encrypted_count} fields")
    if failed_count > 0:
        print(f"⚠️  Skipped (omitted): {failed_count} fields")
    print(f"   Input:  {input_file.resolve()}")
    print(f"   Output: {output_file.resolve()}")
    print("\n⚠️  CRITICAL NEXT STEPS:")
    print(f"   1. Verify output: diff {input_file.name} {output_file.name}")
    print(f"   2. NEVER commit {input_file.name} to Git")
    print(f"   3. Add to .gitignore: echo '{input_file.name}' >> .gitignore")
    print(f"   4. Deploy {output_file.name} as .env to production")


def interactive_encrypt() -> None:
    """交互式加密单个值"""
    print("=" * 60)
    print("🔐 .env Sensitive Value Encryptor")
    print("=" * 60)
    print("\nEnter the environment variable name (e.g., DB_PASSWORD):")

    try:
        key = input("> ").strip()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)

    if not key:
        print("❌ Empty key", file=sys.stderr)
        sys.exit(1)

    print(f"\nEnter the value for {key}:")
    try:
        value = getpass.getpass("> ").strip()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)

    if not value:
        print("❌ Empty value", file=sys.stderr)
        sys.exit(1)

    try:
        encrypted = encrypt_value(value)
    except Exception as e:
        print(f"❌ Encryption failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ ENCRYPTED VALUE")
    print("=" * 60)
    print(f"\n{key}={encrypted}\n")
    print("=" * 60)
    print("\n📋 Copy-paste this line into your .env file")
    print("\n⚠️  SECURITY REMINDERS:")
    print("   • NEVER commit .env with sensitive values to Git")
    print("   • Add .env to .gitignore: echo '.env' >> .gitignore")
    print("   • Rotate keys quarterly using security.rotate_keys()")


def main() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="🔐 Secure .env encryption tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m security.env_encrypt

  # Encrypt single value
  python -m security.env_encrypt DB_PASSWORD

  # Encrypt entire .env file
  python -m security.env_encrypt --encrypt-file .env

  # Decrypt for verification
  python -m security.env_encrypt --decrypt "ENC[gAAAA...]"
        """
    )
    parser.add_argument('key', nargs='?', help="Environment variable name (interactive mode)")
    parser.add_argument('--encrypt-file', metavar='PATH', help="Encrypt sensitive fields in .env file")
    parser.add_argument('--decrypt', metavar='ENC_VALUE', help="Decrypt an ENC[...] value for verification")
    parser.add_argument('--output', metavar='PATH', help="Output file for --encrypt-file")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    args = parser.parse_args()

    try:
        if args.decrypt:
            decrypted = decrypt_value(args.decrypt)
            print(f"Decrypted value: {decrypted}")
        elif args.encrypt_file:
            process_env_file(args.encrypt_file, args.output)
        elif args.key:
            print(f"Encrypting value for {args.key}...")
            try:
                value = input("Value: ").strip()
            except KeyboardInterrupt:
                print("\n\n❌ Operation cancelled by user")
                sys.exit(1)

            if not value:
                print("❌ Empty value", file=sys.stderr)
                sys.exit(1)

            encrypted = encrypt_value(value)
            print(f"{args.key}={encrypted}")
        else:
            interactive_encrypt()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
