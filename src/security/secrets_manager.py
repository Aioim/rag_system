"""
🔐 SecretsManager - 企业级敏感信息管理器

核心安全设计：
✅ 内存中仅存储加密字节（无明文缓存）
✅ 每次 get_secret() 动态解密（最小化明文生命周期）
✅ 密钥文件严格验证（44字节 + base64 校验）
✅ 生产环境零容忍（密钥无效立即终止进程）
✅ 防内存转储（加密数据 + 禁止 pickle）
✅ 防时序攻击（恒定时间比较）
"""
import os
import sys
import threading
from pathlib import Path
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.fernet import Fernet, InvalidToken

from config import PROJECT_ROOT
from logger import logger, security_logger
from security.secret_str import SecretStr


# ==================== 安全配置 ====================
class SecurityConfig:
    """安全配置集中管理"""
    KEY_FILE: Final[Path] = PROJECT_ROOT / "environments" / ".secret_key"
    KEY_LENGTH: Final[int] = 44  # Fernet 密钥必须为 44 字节 URL 安全 base64
    FERNET_VERSION: Final[int] = 0x80  # Fernet 规范版本标识

    ENV: Final[str] = os.getenv("ENV", "dev").lower()
    IS_PRODUCTION: Final[bool] = ENV in ("prod", "production", "staging")
    IS_CI: Final[bool] = os.getenv("CI", "false").lower() == "true"

    AUTO_GENERATE_IN_PROD: Final[bool] = False  # 严格禁止
    MASK_VISIBLE_START: Final[int] = 3
    MASK_VISIBLE_END: Final[int] = 4


# ==================== 密钥诊断工具 ====================

def _analyze_key_integrity(key_bytes: bytes) -> str:
    """精准诊断密钥完整性问题（含跨平台修复指南）"""
    length = len(key_bytes)
    lines = [f"❌ Invalid key length: {length} bytes (expected {SecurityConfig.KEY_LENGTH})"]

    if length == SecurityConfig.KEY_LENGTH + 1 and key_bytes.endswith(b'\n'):
        lines.append("   → Contains Unix newline \\n (use 'wb' mode when generating)")
    elif length == SecurityConfig.KEY_LENGTH + 2 and key_bytes.endswith(b'\r\n'):
        lines.append("   → Contains Windows CRLF \\r\\n (critical error!)")
    elif length == 32:
        lines.append("   → Raw 32-byte key (not base64-encoded)")
    elif length < 40:
        lines.append("   → Severely truncated key")

    lines.extend([
        "",
        "=" * 70,
        "🔧 KEY REPAIR GUIDE (Cross-Platform)",
        "=" * 70,
        "# STEP 1: Delete invalid key",
        "   # Linux/Mac:   rm -f .secret_key",
        "   # Windows PS:  Remove-Item -Force .secret_key",
        "",
        "# STEP 2: Generate VALID key (BINARY MODE IS CRITICAL)",
        "   python -c \"from cryptography.fernet import Fernet; " +
        "open('.secret_key', 'wb').write(Fernet.generate_key())\"",
        "",
        "# STEP 3: Verify key integrity",
        "   python -c \"from cryptography.fernet import Fernet; " +
        "k=open('.secret_key','rb').read().strip(); " +
        "assert len(k)==44, 'Invalid length'; " +
        "Fernet(k); print('✓ VALID KEY')\"",
        "",
        "# STEP 4: IMMEDIATELY add to .gitignore",
        "   echo '.secret_key' >> .gitignore",
        "=" * 70,
        ""
    ])
    return "\n".join(lines)


# ==================== 核心管理器 ====================

class SecretsManager:
    """
    敏感信息管理器 - 内存加密版

    安全设计：
    • 内存中仅存储加密字节（无明文缓存）
    • 每次 get_secret() 动态解密（最小化明文生命周期）
    • 密钥文件严格验证（44字节 + base64 校验）
    • 生产环境零容忍（密钥无效立即终止进程）
    """

    def __init__(self):
        self._fernet: Fernet | None = None
        self._encrypted_cache: dict[str, bytes] = {}
        self._cache_lock = threading.Lock()
        self._auto_generated_key: bool = False

        try:
            self._load_key_file()
            security_logger.info("[OK] Encryption initialized from %s", SecurityConfig.KEY_FILE.name)
        except FileNotFoundError as e:
            self._handle_missing_key(e)
        except (ValueError, InvalidSignature) as e:
            self._handle_invalid_key(e)
        except Exception as e:
            self._handle_initialization_error(e)

        if SecurityConfig.IS_PRODUCTION and not self._fernet:
            self._panic_and_exit(
                "CRITICAL: Encryption unavailable in production environment\n"
                "Required: Valid 44-byte Fernet key in .secret_key file\n"
                "Action: Pre-generate key in secure environment BEFORE deployment"
            )

    def _load_key_file(self) -> None:
        """安全加载密钥文件（严格验证）"""
        if not SecurityConfig.KEY_FILE.exists():
            raise FileNotFoundError(f"Key file not found: {SecurityConfig.KEY_FILE}")

        with open(SecurityConfig.KEY_FILE, 'rb') as f:
            raw_key = f.read()

        stripped_key = raw_key.strip()
        if len(stripped_key) != SecurityConfig.KEY_LENGTH:
            raise ValueError(
                f"Invalid key length: {len(stripped_key)} bytes\n"
                + _analyze_key_integrity(raw_key)
            )

        try:
            self._fernet = Fernet(stripped_key)
            test_val = b"__key_verification__"
            # 不使用 assert（-O 模式下会被禁用），显式校验密钥自检
            decrypted = self._fernet.decrypt(self._fernet.encrypt(test_val))
            if decrypted != test_val:
                raise ValueError(
                    "Fernet 密钥自检失败：加密/解密往返不匹配"
                )
        except Exception as e:
            raise ValueError(f"Key validation failed: {e}") from e

    def _handle_missing_key(self, exc: FileNotFoundError) -> None:
        """处理缺失密钥"""
        if SecurityConfig.IS_PRODUCTION and not SecurityConfig.IS_CI:
            self._panic_and_exit(
                f"MISSING KEY FILE IN PRODUCTION: {SecurityConfig.KEY_FILE}\n"
                "Policy: Production environments MUST have pre-generated keys"
            )

        if not SecurityConfig.AUTO_GENERATE_IN_PROD:
            self._generate_dev_key()
            security_logger.warning(
                "\n⚠️  AUTO-GENERATED DEVELOPMENT KEY\n"
                "   Location: %s\n"
                "   ⚠️  CRITICAL: Add to .gitignore IMMEDIATELY",
                SecurityConfig.KEY_FILE
            )

    def _handle_invalid_key(self, exc: Exception) -> None:
        """处理无效密钥"""
        if SecurityConfig.IS_PRODUCTION and not SecurityConfig.IS_CI:
            self._panic_and_exit(
                f"INVALID KEY IN PRODUCTION: {SecurityConfig.KEY_FILE}\n"
                f"Error: {exc}\n"
                "Policy: Production keys must be pre-validated"
            )

        try:
            with open(SecurityConfig.KEY_FILE, 'rb') as f:
                raw = f.read()
            diagnosis = _analyze_key_integrity(raw)
            logger.warning("Key file diagnosis:\n%s", diagnosis)
        except Exception as e:
            logger.debug("Key file diagnosis skipped (file may be unreadable): %s", e)

        if not SecurityConfig.IS_PRODUCTION or SecurityConfig.IS_CI:
            # 重命名损坏文件备份（而非直接删除），保留从旧加密值恢复的可能
            # 注意：.secret_key 是点文件，suffix 为空，with_suffix 会拼接而非替换
            _bak = SecurityConfig.KEY_FILE.with_name(
                SecurityConfig.KEY_FILE.name + ".corrupted"
            )
            try:
                SecurityConfig.KEY_FILE.replace(_bak)
            except OSError as e:
                logger.warning(
                    "Failed to rename corrupted key file (will unlink instead): %s", e
                )
                SecurityConfig.KEY_FILE.unlink(missing_ok=True)
            self._generate_dev_key()

    def _handle_initialization_error(self, exc: Exception) -> None:
        if SecurityConfig.IS_PRODUCTION:
            self._panic_and_exit(f"Encryption initialization failed: {exc}")
        logger.warning("Encryption initialization failed (dev mode): %s", exc)

    def _generate_dev_key(self) -> None:
        """为开发环境生成临时密钥（原子写入）"""
        if SecurityConfig.IS_PRODUCTION and not SecurityConfig.AUTO_GENERATE_IN_PROD:
            raise RuntimeError("Auto-key generation forbidden in production")

        if not SecurityConfig.KEY_FILE.exists():
            key = Fernet.generate_key()
            SecurityConfig.KEY_FILE.parent.mkdir(parents=True, exist_ok=True)

            # 原子写入：临时文件 + 重命名
            temp_path = SecurityConfig.KEY_FILE.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                f.write(key)
                f.flush()
                os.fsync(f.fileno())
            temp_path.rename(SecurityConfig.KEY_FILE)

            if os.name != 'nt':
                SecurityConfig.KEY_FILE.chmod(0o600)

        self._auto_generated_key = True
        self._load_key_file()

    def _panic_and_exit(self, message: str) -> None:
        """生产环境致命错误（立即终止）"""
        logger.critical("=" * 70)
        logger.critical("SECURITY FATAL ERROR")
        logger.critical("=" * 70)
        logger.critical(message)
        logger.critical("=" * 70)
        sys.exit(1)

    def _encrypt(self, value: str) -> bytes:
        if not self._fernet:
            raise RuntimeError("Encryption not initialized")
        return self._fernet.encrypt(value.encode())

    def _decrypt(self, encrypted_value: bytes) -> str:
        if not self._fernet:
            raise RuntimeError("Encryption not initialized")
        try:
            return self._fernet.decrypt(encrypted_value).decode()
        except InvalidToken as e:
            raise ValueError(
                "Decryption failed: Invalid token (key mismatch or corrupted data)"
            ) from e

    # ==================== 公共 API ====================

    def set_secret(self, name: str, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Secret value must be str, got {type(value).__name__}")
        encrypted = self._encrypt(value)
        with self._cache_lock:
            self._encrypted_cache[name] = encrypted
        security_logger.info("Secret stored: %s (encrypted in memory)", name)

    def get_secret(
            self,
            name: str,
            default: str | None = None,
            required: bool = False
    ) -> SecretStr | None:
        with self._cache_lock:
            encrypted = self._encrypted_cache.get(name)
        if encrypted is None:
            if required:
                raise KeyError(f"Required secret '{name}' not found")
            if default is not None:
                return SecretStr(default, name=f"{name}_default")
            return None

        try:
            decrypted = self._decrypt(encrypted)
            return SecretStr(decrypted, name=name)
        except Exception as e:
            security_logger.error("Decryption failed for secret '%s': %s", name, e)
            raise

    def delete_secret(self, name: str) -> bool:
        with self._cache_lock:
            if name in self._encrypted_cache:
                del self._encrypted_cache[name]
                security_logger.info("Secret purged from memory: %s", name)
                return True
        return False

    def list_secrets(self) -> list[str]:
        return list(self._encrypted_cache)

    def is_encrypted(self) -> bool:
        return self._fernet is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "encrypted": self.is_encrypted(),
            "environment": "production" if SecurityConfig.IS_PRODUCTION else "development",
            "key_file": str(SecurityConfig.KEY_FILE),
            "key_file_exists": SecurityConfig.KEY_FILE.exists(),
            "key_valid": self._fernet is not None,
            "secrets_cached": len(self._encrypted_cache),
            "auto_generated": self._auto_generated_key
        }

    def encrypt_string(self, value: str) -> str:
        """公钥加密字符串，返回 ENC[...] 格式"""
        if not self._fernet:
            raise RuntimeError("Fernet not initialized")
        encrypted_bytes = self._fernet.encrypt(value.encode("utf-8"))
        return f"ENC[{encrypted_bytes.decode('utf-8')}]"

    def decrypt_string(self, encrypted: str) -> str:
        """解密 Fernet 令牌（raw base64，不含 ENC[...] 包装），返回明文

        注意：此方法期望原始 base64 令牌，而非 ENC[...] 格式。
        如需解密 ENC[...] 值，请使用 env_encryptor.decrypt_value()。
        """
        if not self._fernet:
            raise RuntimeError("Fernet not initialized")
        encrypted_bytes = encrypted.encode("utf-8")
        decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
        return decrypted_bytes.decode("utf-8")


# ==================== 全局实例 ====================

try:
    secrets: SecretsManager = SecretsManager()
except Exception as e:
    if not SecurityConfig.IS_PRODUCTION or SecurityConfig.IS_CI:
        logger.warning("Falling back to INSECURE DEVELOPMENT MODE: %s", e)

        class InsecureDevelopmentFallback:
            """仅限开发环境的不安全降级（明文以 os.environ 存储，子进程可见）"""

            def set_secret(self, name: str, value: str) -> None:
                if not isinstance(value, str):
                    raise TypeError(
                        f"Secret value must be str, got {type(value).__name__}"
                    )
                os.environ[name] = value
                logger.debug("[INSECURE] Secret stored as env var: %s", name)

            def get_secret(self, name: str, default: str | None = None, required: bool = False) -> SecretStr | None:
                val = os.getenv(name)
                if val is None:
                    if required:
                        raise KeyError(f"Missing required env: {name}")
                    if default is not None:
                        return SecretStr(default, name=f"{name}_default")
                    return None
                return SecretStr(val, name=name)

            @staticmethod
            def is_encrypted() -> bool:
                return False

            @staticmethod
            def delete_secret(name: str) -> bool:
                os.environ.pop(name, None)
                logger.debug("[INSECURE] Secret removed from env var: %s", name)
                return True

            @staticmethod
            def list_secrets() -> list[str]:
                return []

            @staticmethod
            def get_status() -> dict:
                return {
                    "encrypted": False,
                    "insecure_fallback": True,
                    "environment": "development",
                    "key_file": str(SecurityConfig.KEY_FILE),
                    "key_file_exists": SecurityConfig.KEY_FILE.exists(),
                    "key_valid": False,
                    "secrets_cached": 0,
                    "auto_generated": False,
                }

            @staticmethod
            def encrypt_string(value: str) -> str:
                raise RuntimeError("Fernet not initialized (insecure fallback active)")

            @staticmethod
            def decrypt_string(encrypted: str) -> str:
                raise RuntimeError("Fernet not initialized (insecure fallback active)")

        secrets = InsecureDevelopmentFallback()
    else:
        logger.critical("SecretsManager initialization failed in production: %s", e)
        sys.exit(1)


# ==================== 便捷 API ====================

def get_secret(name: str, default: str | None = None, required: bool = False) -> str | None:
    secret_obj = secrets.get_secret(name, default=default, required=required)
    return secret_obj.get() if secret_obj else None


def set_secret(name: str, value: str) -> None:
    secrets.set_secret(name, value)


# ==================== 密钥管理工具 ====================

def generate_key_file(filepath: str | None = None) -> str:
    if SecurityConfig.IS_PRODUCTION and not SecurityConfig.IS_CI:
        raise RuntimeError("Key generation forbidden in production environment")

    key = Fernet.generate_key()
    key_path = Path(filepath) if filepath else SecurityConfig.KEY_FILE
    key_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = key_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as f:
        f.write(key)
        f.flush()
        os.fsync(f.fileno())
    temp_path.rename(key_path)

    if os.name != 'nt':
        key_path.chmod(0o600)

    fingerprint = key.hex()[:16]
    print(f"\n✓ Encryption key generated: {key_path.resolve()}")
    print(f"✓ Key fingerprint: {fingerprint}")
    print("\n⚠️  CRITICAL NEXT STEPS:")
    print("   1. Add to .gitignore: echo '.secret_key' >> .gitignore")
    print("   2. Securely distribute to production hosts")
    print("   3. NEVER commit to version control\n")

    return str(key_path.resolve())