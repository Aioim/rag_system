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

from .secrets_manager import (
    SecretsManager,
    SecurityConfig,
    secrets,
    get_secret,
    set_secret,
    generate_key_file,
)
from .secret_str import SecretStr
from .secure_env_loader import (
    SecureEnvLoader,
    load_secure_dotenv,
)
from .env_encryptor import (
    encrypt_value,
    decrypt_value,
    fetch_and_decrypt_env_var,
    process_env_file,
)

__all__ = [
    "SecretsManager",
    "SecurityConfig",
    "secrets",
    "get_secret",
    "set_secret",
    "generate_key_file",
    "SecretStr",
    "SecureEnvLoader",
    "load_secure_dotenv",
    "encrypt_value",
    "decrypt_value",
    "fetch_and_decrypt_env_var",
    "process_env_file",
    "__version__",
]