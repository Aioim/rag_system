"""SecretsManager 和便捷函数单元测试"""

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from security import SecretStr
from security.secrets_manager import (
    InsecureDevelopmentFallback,
    SecurityConfig,
    SecretsManager,
    _analyze_key_integrity,
    generate_key_file,
    get_secret,
    set_secret,
)


class TestSecurityConfig:
    def test_key_length_is_44(self):
        assert SecurityConfig.KEY_LENGTH == 44

    def test_fernet_version(self):
        assert SecurityConfig.FERNET_VERSION == 0x80


class TestAnalyzeKeyIntegrity:
    def test_too_short(self):
        result = _analyze_key_integrity(b"short")
        assert "Invalid key length" in result

    def test_unix_newline(self):
        key = Fernet.generate_key() + b"\n"
        result = _analyze_key_integrity(key)
        assert "Unix newline" in result

    def test_windows_crlf(self):
        key = Fernet.generate_key() + b"\r\n"
        result = _analyze_key_integrity(key)
        assert "Windows CRLF" in result

    def test_raw_32_bytes(self):
        result = _analyze_key_integrity(b"a" * 32)
        assert "Raw 32-byte key" in result


class TestInsecureDevelopmentFallback:
    """测试独立的降级类（验证提取到模块级别后可独立测试）"""

    @pytest.fixture
    def fallback(self):
        return InsecureDevelopmentFallback()

    def test_set_and_get_secret(self, fallback, clean_env):
        fallback.set_secret("TEST_KEY", "test_value")
        result = fallback.get_secret("TEST_KEY")
        assert result is not None
        assert result.get() == "test_value"

    def test_get_secret_default(self, fallback, clean_env):
        result = fallback.get_secret("NONEXISTENT", default="fallback_val")
        assert result is not None
        assert result.get() == "fallback_val"

    def test_get_secret_required_raises(self, fallback, clean_env):
        # 确保环境变量不存在
        os.environ.pop("NONEXISTENT", None)
        with pytest.raises(KeyError, match="Missing required env"):
            fallback.get_secret("NONEXISTENT", required=True)

    def test_get_secret_none(self, fallback, clean_env):
        os.environ.pop("NONEXISTENT", None)
        result = fallback.get_secret("NONEXISTENT")
        assert result is None

    def test_set_secret_rejects_non_string(self, fallback):
        with pytest.raises(TypeError, match="must be str"):
            fallback.set_secret("KEY", 123)  # type: ignore[arg-type]

    def test_is_encrypted_always_false(self, fallback):
        assert not fallback.is_encrypted()

    def test_delete_secret(self, fallback, clean_env):
        fallback.set_secret("DEL_KEY", "val")
        assert fallback.delete_secret("DEL_KEY")
        assert os.getenv("DEL_KEY") is None

    def test_delete_nonexistent(self, fallback, clean_env):
        assert fallback.delete_secret("NOT_THERE")

    def test_list_secrets_always_empty(self, fallback):
        fallback.set_secret("K", "v")
        assert fallback.list_secrets() == []

    def test_get_status(self, fallback):
        status = fallback.get_status()
        assert status["encrypted"] is False
        assert status["insecure_fallback"] is True
        assert "environment" in status

    def test_encrypt_string_raises(self, fallback):
        with pytest.raises(RuntimeError, match="insecure fallback"):
            fallback.encrypt_string("val")

    def test_decrypt_string_raises(self, fallback):
        with pytest.raises(RuntimeError, match="insecure fallback"):
            fallback.decrypt_string("enc")


class TestSecretsManagerWithKey:
    """使用真实密钥文件测试 SecretsManager"""

    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        """创建带临时密钥的 SecretsManager"""
        key = Fernet.generate_key()
        key_file = tmp_path / ".secret_key"
        key_file.write_bytes(key)

        # 临时覆盖 SecurityConfig.KEY_FILE
        monkeypatch.setattr(SecurityConfig, "KEY_FILE", key_file)
        monkeypatch.setattr(SecurityConfig, "IS_PRODUCTION", False)
        monkeypatch.setattr(SecurityConfig, "IS_CI", False)

        return SecretsManager()

    def test_initialization_success(self, manager):
        assert manager.is_encrypted()

    def test_set_and_get_secret(self, manager):
        manager.set_secret("api_key", "sk-test-123")
        result = manager.get_secret("api_key")
        assert result is not None
        assert result.get() == "sk-test-123"
        assert isinstance(result, SecretStr)

    def test_get_secret_nonexistent_default(self, manager):
        result = manager.get_secret("not_there", default="default_val")
        assert result is not None
        assert result.get() == "default_val"

    def test_get_secret_nonexistent_none(self, manager):
        result = manager.get_secret("not_there")
        assert result is None

    def test_get_secret_required_raises(self, manager):
        with pytest.raises(KeyError, match="Required secret"):
            manager.get_secret("not_there", required=True)

    def test_set_secret_rejects_non_string(self, manager):
        with pytest.raises(TypeError, match="must be str"):
            manager.set_secret("key", 123)  # type: ignore[arg-type]

    def test_delete_secret(self, manager):
        manager.set_secret("temp", "val")
        assert manager.delete_secret("temp")
        assert manager.get_secret("temp") is None

    def test_delete_nonexistent(self, manager):
        assert not manager.delete_secret("not_there")

    def test_list_secrets(self, manager):
        manager.set_secret("a", "1")
        manager.set_secret("b", "2")
        secrets_list = manager.list_secrets()
        assert "a" in secrets_list
        assert "b" in secrets_list

    def test_get_status(self, manager):
        status = manager.get_status()
        assert status["encrypted"] is True
        assert status["key_valid"] is True
        assert isinstance(status["secrets_cached"], int)

    def test_encrypt_string(self, manager):
        result = manager.encrypt_string("hello")
        assert result.startswith("ENC[")
        assert result.endswith("]")

    def test_encrypt_decrypt_roundtrip(self, manager):
        encrypted = manager.encrypt_string("secret_value")
        # 提取 ENC[...] 中的 base64 部分
        inner = encrypted[4:-1]
        decrypted = manager.decrypt_string(inner)
        assert decrypted == "secret_value"

    def test_decrypt_invalid_token(self, manager):
        """decrypt_string 对无效 base64 直接抛出 cryptography.fernet.InvalidToken"""
        from cryptography.fernet import InvalidToken

        with pytest.raises(InvalidToken):
            manager.decrypt_string("invalid_base64_token")

    def test_encrypt_string_uninitialized(self):
        """未初始化 Fernet 时 encrypt_string 抛出异常"""
        fallback = InsecureDevelopmentFallback()
        with pytest.raises(RuntimeError, match="insecure fallback"):
            fallback.encrypt_string("val")

    def test_concurrent_access(self, manager):
        """验证多线程并发 set/get 不抛出异常"""
        errors = []

        def worker(i):
            try:
                manager.set_secret(f"key_{i}", f"val_{i}")
                result = manager.get_secret(f"key_{i}")
                assert result is not None
                assert result.get() == f"val_{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发访问出现错误: {errors}"


class TestModuleLevelConvenienceFunctions:
    """测试模块级便捷函数（需要已初始化的 secrets 单例）"""

    def test_get_secret_returns_secret_str(self, monkeypatch):
        """验证便捷函数返回 SecretStr（而非裸 str）"""
        from security import secrets

        secrets.set_secret("conv_test", "conv_val")
        result = get_secret("conv_test")
        assert isinstance(result, SecretStr)
        assert result.get() == "conv_val"

    def test_get_secret_nonexistent(self):
        result = get_secret("nonexistent_xyz")
        assert result is None

    def test_set_secret(self):
        set_secret("module_level", "module_val")
        result = get_secret("module_level")
        assert result is not None
        assert result.get() == "module_val"


class TestGenerateKeyFile:
    def test_generates_valid_key(self, tmp_path, monkeypatch):
        """生成密钥文件并验证格式"""
        monkeypatch.setattr(SecurityConfig, "IS_PRODUCTION", False)
        monkeypatch.setattr(SecurityConfig, "IS_CI", False)

        key_path = tmp_path / "generated.key"
        result = generate_key_file(str(key_path))
        assert result == str(key_path.resolve())
        assert key_path.exists()

        # 验证密钥可被 Fernet 加载
        with open(key_path, "rb") as f:
            key_bytes = f.read().strip()
        assert len(key_bytes) == 44
        Fernet(key_bytes)  # 不抛异常即有效
