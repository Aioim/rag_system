"""
🔐 Security Module - 敏感信息管理套件

核心组件：
- SecretsManager: 内存加密存储与解密
- SecureEnvLoader: 安全的 .env 文件加载器
- SecretStr: 防泄露的敏感字符串容器

使用示例：
    from security import secrets, load_secure_dotenv, SecretStr

    load_secure_dotenv()                        # 安全加载 .env，自动解密 ENC[...]
    secrets.set_secret("api_key", "sk-xxx")     # 内存加密存储
    api_key = secrets.get_secret("api_key")     # 动态解密
"""

__version__ = "2.0.0"

from .env_encryptor import (
    decrypt_value,
    encrypt_value,
    fetch_and_decrypt_env_var,
    process_env_file,
)
from .secret_str import SecretStr
from .secrets_manager import (
    SecretsManager,
    SecurityConfig,
    generate_key_file,
    get_secret,
    secrets,
    set_secret,
)
from .secure_env_loader import (
    SecureEnvLoader,
    load_secure_dotenv,
)

__all__ = [
    "SecretStr",
    "SecretsManager",
    "SecureEnvLoader",
    "SecurityConfig",
    "__version__",
    "decrypt_value",
    "encrypt_value",
    "fetch_and_decrypt_env_var",
    "generate_key_file",
    "get_secret",
    "load_secure_dotenv",
    "process_env_file",
    "secrets",
    "set_secret",
]