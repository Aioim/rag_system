"""
🔐 SecureEnvLoader - 安全的 .env 文件加载器

特性：
- 自动识别 ENC[...] 加密字段
- 与 python-dotenv 完全兼容
- 解密失败时提供精准诊断
- 防止敏感字段意外泄露到日志
- 支持多行值、引号、转义字符
"""
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
from cryptography.fernet import InvalidToken

from security.secrets_manager import SecretsManager
from security.secret_str import SecretStr
from logger import security_logger, logger


class SecureEnvLoader:
    """安全的环境变量加载器"""

    ENC_PATTERN = re.compile(r'^ENC\[(?P<value>[A-Za-z0-9_\-+=/]+)\]$')
    MAX_ENC_LENGTH = 1024  # Fernet token 通常小于 200 字节

    SECRET_KEY_PATTERNS: Set[str] = {
        'password', 'pwd', 'secret', 'key', 'token', 'credential',
        'api_key', 'access_key', 'secret_key', 'webhook_secret',
        'private_key', 'cert', 'certificate'
    }

    def __init__(self, env_file: Optional[Path] = None):
        self.env_file = env_file or Path('.env')
        self.secrets_manager = SecretsManager()
        self._loaded_values: Dict[str, str] = {}
        self._decryption_errors: Dict[str, str] = {}

    def load(self, override: bool = False) -> Dict[str, str]:
        if not self.env_file.exists():
            logger.warning("Env file not found: %s", self.env_file)
            return {}

        raw_lines = self._read_env_lines()
        parsed = self._parse_env_lines(raw_lines)

        encrypted_fields = {}
        plain_fields = {}

        for key, value in parsed.items():
            if self.is_encrypted_value(value):
                match = self.ENC_PATTERN.match(value)
                encrypted_fields[key] = match.group('value')
            else:
                plain_fields[key] = value

        for key, value in plain_fields.items():
            if not override and key in os.environ:
                continue
            os.environ[key] = value
            self._loaded_values[key] = self._redact_value(key, value)

        for key, encrypted_value in encrypted_fields.items():
            try:
                decrypted = self._decrypt_env_value(encrypted_value)
                if not override and key in os.environ:
                    continue
                os.environ[key] = decrypted
                self._loaded_values[key] = self._redact_value(key, decrypted)
                security_logger.info("Decrypted sensitive env var: %s", key)
            except Exception as e:
                self._decryption_errors[key] = str(e)
                security_logger.error("Decryption failed for %s: %s", key, e)

        self._log_load_summary(plain_fields, encrypted_fields)

        if self._is_production() and self._decryption_errors:
            self._fatal_error(
                f"CRITICAL: Failed to decrypt {len(self._decryption_errors)} sensitive fields\n"
                + "\n".join(f"  • {k}: {v}" for k, v in self._decryption_errors.items())
            )

        return self._loaded_values

    def _read_env_lines(self) -> list[Tuple[int, str]]:
        lines = []
        with open(self.env_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                lines.append((idx, line.rstrip('\n\r')))
        return lines

    def _parse_env_lines(self, lines: list[Tuple[int, str]]) -> Dict[str, str]:
        env_vars = {}
        current_key = None
        current_value = []
        in_quotes = False
        quote_char = None

        for line_no, line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            if '=' in stripped and not in_quotes:
                key, value = stripped.split('=', 1)
                key = key.strip()
                value = value.strip()

                if value.startswith('"') and not value.endswith('"'):
                    in_quotes = True
                    quote_char = '"'
                    current_key = key
                    current_value = [value[1:]]
                    continue
                elif value.startswith("'") and not value.endswith("'"):
                    in_quotes = True
                    quote_char = "'"
                    current_key = key
                    current_value = [value[1:]]
                    continue
                else:
                    env_vars[key] = self._unescape_value(value)
            elif in_quotes:
                if stripped.endswith(quote_char):
                    current_value.append(stripped[:-1])
                    env_vars[current_key] = '\n'.join(current_value)
                    in_quotes = False
                    current_key = None
                    current_value = []
                    quote_char = None
                else:
                    current_value.append(stripped)

        return env_vars

    def _unescape_value(self, value: str) -> str:
        value = value.strip().strip('"').strip("'")
        return value.replace('\\\\', '\\').replace('\\n', '\n').replace('\\t', '\t')

    def _decrypt_env_value(self, encrypted_b64: str) -> str:
        encrypted_bytes = encrypted_b64.encode('utf-8')
        if not self.secrets_manager._fernet:
            raise RuntimeError("Fernet not initialized")
        try:
            decrypted_bytes = self.secrets_manager._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except InvalidToken as e:
            raise ValueError(
                "Decryption failed: Invalid token (key mismatch or corrupted data).\n"
                "Common causes:\n"
                "  • .secret_key was regenerated after encrypting this value\n"
                "  • Value was manually edited (base64 corruption)\n"
                "  • Using wrong environment's .secret_key"
            ) from e

    def _redact_value(self, key: str, value: str) -> str:
        key_lower = key.lower()
        # 完全掩码（密码类）
        if any(k in key_lower for k in ['password', 'pwd', 'secret', 'token']):
            return "******"
        # 部分掩码（API密钥）
        if any(k in key_lower for k in ['key', 'api']):
            if len(value) > 8:
                return f"{value[:4]}...{value[-4:]}"
            return "******"
        # 邮箱掩码（优先检测值特征）
        if '@' in value:
            try:
                local, domain = value.split('@', 1)
                if len(local) > 1:
                    return f"{local[0]}***@{domain}"
            except ValueError:
                pass
        # 默认截断
        return value if len(value) < 20 else f"{value[:15]}..."

    def _log_load_summary(self, plain_fields: Dict, encrypted_fields: Dict):
        total = len(plain_fields) + len(encrypted_fields)
        failed = len(self._decryption_errors)
        success_enc = len(encrypted_fields) - failed

        security_logger.info(
            "Env loaded: %d total (%d plain, %d encrypted, %d failed)",
            total, len(plain_fields), len(encrypted_fields), failed
        )

        if encrypted_fields:
            logger.info("✓ Loaded %d encrypted secrets (e.g., DB_PASSWORD=******)", success_enc)

        if self._decryption_errors:
            logger.warning(
                "⚠️  Failed to decrypt %d fields: %s",
                failed,
                ", ".join(self._decryption_errors.keys())
            )

    def _is_production(self) -> bool:
        return os.getenv("ENV", "dev").lower() in ("prod", "production", "staging")

    def _fatal_error(self, message: str):
        logger.critical("=" * 70)
        logger.critical("SECURITY FAILURE: Env decryption failed in production")
        logger.critical("=" * 70)
        logger.critical(message)
        logger.critical("=" * 70)
        sys.exit(1)

    @classmethod
    def is_encrypted_value(cls, value: str) -> bool:
        if len(value) > cls.MAX_ENC_LENGTH:
            return False
        return bool(cls.ENC_PATTERN.match(value))


# ========== 全局便捷函数 ==========

def load_secure_dotenv(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    env_path = Path(dotenv_path) if dotenv_path else Path('.env')
    loader = SecureEnvLoader(env_path)
    loader.load(override=override)
    return True


if __name__ == "__main__":
    load_secure_dotenv()