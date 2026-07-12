# Security 模块 — 敏感信息管理

## 模块概述

Security 模块提供内存加密存储、安全 .env 加载、敏感数据脱敏等能力。

- **内存加密**：敏感信息仅在内存中加密存储
- **防泄露**：自动脱敏、禁止序列化、恒定时间比较
- **ENC 解密**：自动识别并解密 `.env` 文件中的 `ENC[...]` 字段

## 文件结构

```
security/
├── __init__.py            # 导出
├── secret_str.py          # SecretStr — 敏感字符串容器
├── secrets_manager.py     # SecretsManager — 内存加密存储
├── secure_env_loader.py   # SecureEnvLoader — .env + ENC 解密
├── env_encryptor.py       # CLI 加密工具
└── README.md
```

## 快速开始

```python
from security import secrets, load_secure_dotenv, SecretStr

# 安全加载 .env（自动解密 ENC[...] 字段）
load_secure_dotenv()

# 内存加密存储
secrets.set_secret("api_key", "sk-xxx")
key = secrets.get_secret("api_key")   # → SecretStr('***')
key.get()                              # → 'sk-xxx'
```

## 核心组件

### SecretStr — 敏感字符串容器

```python
from security import SecretStr

pwd = SecretStr("my_password", name="db")
print(pwd)           # my********rd
pwd.get()            # my_password
pwd.mask()           # my********rd
```

防护措施：禁止打印、禁止序列化、恒定时间比较、`__del__` 内存清零。

### SecretsManager — 内存加密管理器

```python
from security import secrets, get_secret, set_secret

set_secret("db_password", "secure123")
val = get_secret("db_password")          # 动态解密
secrets.list_secrets()                    # ['db_password']
secrets.delete_secret("db_password")
```

内存中仅存储 Fernet 加密字节，每次 `get_secret()` 动态解密。

### SecureEnvLoader — 安全 .env 加载

```python
from security import load_secure_dotenv

load_secure_dotenv()                      # 加载 .env
# .env 中的 ENC[...] 字段自动解密
```

支持 `.env` 格式：多行值、引号、转义字符。

### CLI 加密工具

```bash
# 交互式加密
python -m security.env_encrypt

# 加密单个值
python -m security.env_encrypt DB_PASSWORD

# 批量加密 .env 文件
python -m security.env_encrypt --encrypt-file .env

# 解密验证
python -m security.env_encrypt --decrypt "ENC[gAAAA...]"
```

### 编程接口

```python
from security import encrypt_value, decrypt_value, process_env_file

encrypted = encrypt_value("my_secret")    # → ENC[...]
decrypt_value(encrypted)                   # → my_secret
process_env_file(".env", ".env.encrypted") # 批量加密
```

## 密钥管理

```bash
# 生成密钥
python -c "from cryptography.fernet import Fernet; \
  open('.secret_key', 'wb').write(Fernet.generate_key())"

# 验证
python -c "from cryptography.fernet import Fernet; \
  k=open('.secret_key','rb').read().strip(); Fernet(k); print('OK')"

# .gitignore
echo '.secret_key' >> .gitignore
```

或使用代码：
```python
from security import generate_key_file
generate_key_file()
```

## API 参考

| 类/函数 | 说明 |
|---------|------|
| `SecretStr(value, name)` | 敏感字符串容器 |
| `secrets` | SecretsManager 全局单例 |
| `get_secret(name)` | 获取并解密敏感值 |
| `set_secret(name, value)` | 加密存储敏感值 |
| `load_secure_dotenv(path)` | 安全加载 .env 文件 |
| `encrypt_value(value)` | 加密为 `ENC[...]` 格式 |
| `decrypt_value(enc_str)` | 解密 `ENC[...]` 格式 |
| `process_env_file(in, out)` | 批量加密 .env 敏感字段 |
| `fetch_and_decrypt_env_var(name)` | 读取并解密环境变量 |
| `generate_key_file(path)` | 生成 Fernet 密钥文件 |

## 依赖

```bash
pip install cryptography python-dotenv
```
