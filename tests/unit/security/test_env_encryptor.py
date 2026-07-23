"""env_encryptor 单元测试"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from security import decrypt_value, encrypt_value, fetch_and_decrypt_env_var, process_env_file
from security.secrets_manager import InsecureDevelopmentFallback, SecurityConfig
from security.secure_env_loader import SecureEnvLoader


class TestEncryptDecryptValue:
    @pytest.fixture(autouse=True)
    def _setup_secrets(self, monkeypatch):
        """确保 secrets 单例中有有效的 Fernet 或降级，让 encrypt_value 可用"""
        from security import secrets as _global_secrets
        # 如果 secrets 不是 SecretsManager 实例（降级场景），需要 Fernet
        if isinstance(_global_secrets, InsecureDevelopmentFallback):
            pytest.skip("secrets 已降级为 InsecureDevelopmentFallback，encrypt/decrypt 不可用")

    def test_encrypt_value_returns_enc_format(self):
        result = encrypt_value("test_secret")
        assert result.startswith("ENC[")
        assert result.endswith("]")
        assert "test_secret" not in result

    def test_encrypt_decrypt_roundtrip(self):
        encrypted = encrypt_value("my_password_123")
        decrypted = decrypt_value(encrypted)
        assert decrypted == "my_password_123"

    def test_decrypt_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid ENC format"):
            decrypt_value("not_enc_format")

    def test_decrypt_no_brackets(self):
        with pytest.raises(ValueError, match="Invalid ENC format"):
            decrypt_value("ENC[incomplete")

    def test_encrypt_different_values_produce_different_ciphertexts(self):
        """Fernet 每次加密产生不同密文（含时间戳）"""
        r1 = encrypt_value("same_value")
        r2 = encrypt_value("same_value")
        assert r1 != r2  # 时间戳不同导致密文不同


class TestFetchAndDecryptEnvVar:
    def test_fetch_existing_env_var(self, monkeypatch):
        """从环境变量中获取并解密 ENC 值"""
        encrypted = encrypt_value("secure_pass")
        monkeypatch.setenv("TEST_PASSWORD", encrypted)

        result = fetch_and_decrypt_env_var("TEST_PASSWORD")
        assert result == "secure_pass"

    def test_fetch_nonexistent_var(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        with pytest.raises(ValueError, match="not found"):
            fetch_and_decrypt_env_var("NONEXISTENT_VAR")

    def test_fetch_empty_var(self, monkeypatch):
        monkeypatch.setenv("EMPTY_VAR", "   ")
        with pytest.raises(ValueError, match="is empty"):
            fetch_and_decrypt_env_var("EMPTY_VAR")


class TestProcessEnvFile:
    def test_encrypts_sensitive_fields(self, tmp_path, monkeypatch):
        """敏感字段应被加密"""
        monkeypatch.setattr(SecurityConfig, "IS_PRODUCTION", False)
        monkeypatch.setattr(SecurityConfig, "IS_CI", False)

        input_file = tmp_path / ".env"
        input_file.write_text(
            "PUBLIC_CONFIG=visible\n"
            "DB_PASSWORD=secret123\n"
            "API_KEY=sk-api-key\n",
            encoding="utf-8",
        )
        output_file = tmp_path / ".env.encrypted"

        process_env_file(str(input_file), str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "PUBLIC_CONFIG=visible" in content
        assert "ENC[" in content
        assert "secret123" not in content
        assert "sk-api-key" not in content

    def test_skips_already_encrypted(self, tmp_path, monkeypatch):
        """已加密的字段不重复加密"""
        monkeypatch.setattr(SecurityConfig, "IS_PRODUCTION", False)
        monkeypatch.setattr(SecurityConfig, "IS_CI", False)

        already_enc = encrypt_value("already_encrypted")
        input_file = tmp_path / ".env"
        input_file.write_text(f"DB_PASSWORD={already_enc}\n", encoding="utf-8")
        output_file = tmp_path / ".env.encrypted"

        process_env_file(str(input_file), str(output_file))

        content = output_file.read_text()
        # 不应有 ENC[ 嵌套 ENC[
        enc_count = content.count("ENC[")
        assert enc_count == 1

    def test_skips_comments_and_empty_lines(self, tmp_path, monkeypatch):
        """空行和注释行保持不变"""
        monkeypatch.setattr(SecurityConfig, "IS_PRODUCTION", False)
        monkeypatch.setattr(SecurityConfig, "IS_CI", False)

        input_file = tmp_path / ".env"
        input_file.write_text(
            "# This is a comment\n"
            "\n"
            "DB_PASSWORD=secret\n",
            encoding="utf-8",
        )
        output_file = tmp_path / ".env.encrypted"

        process_env_file(str(input_file), str(output_file))

        content = output_file.read_text()
        assert "# This is a comment" in content
        assert content.count("\n") >= 3  # 保留空行

    def test_failed_encrypt_writes_comment_marker(self, tmp_path, monkeypatch):
        """加密失败时写入注释标记而非静默丢弃"""
        monkeypatch.setattr(SecurityConfig, "IS_PRODUCTION", False)
        monkeypatch.setattr(SecurityConfig, "IS_CI", False)

        input_file = tmp_path / ".env"
        input_file.write_text("DB_PASSWORD=should_be_encrypted\n", encoding="utf-8")
        output_file = tmp_path / ".env.encrypted"

        # 模拟加密失败
        with patch("security.env_encryptor.encrypt_value", side_effect=RuntimeError("mock error")):
            process_env_file(str(input_file), str(output_file))

        content = output_file.read_text()
        assert "# FAILED TO ENCRYPT: DB_PASSWORD=should_be_encrypted" in content

    def test_nonexistent_input_exits(self, tmp_path):
        """输入文件不存在时 sys.exit(1)"""
        with pytest.raises(SystemExit):
            process_env_file(str(tmp_path / "nonexistent.env"))

    def test_non_sensitive_fields_preserved(self, tmp_path, monkeypatch):
        """TIMEZONE / MAX_RETRIES 等字段不被加密"""
        monkeypatch.setattr(SecurityConfig, "IS_PRODUCTION", False)
        monkeypatch.setattr(SecurityConfig, "IS_CI", False)

        input_file = tmp_path / ".env"
        input_file.write_text(
            "TIMEZONE=Asia/Shanghai\n"
            "MAX_RETRIES=3\n"
            "KEYBOARD_LAYOUT=us\n",  # 不应匹配 "key" 子串
            encoding="utf-8",
        )
        output_file = tmp_path / ".env.encrypted"

        process_env_file(str(input_file), str(output_file))

        content = output_file.read_text()
        assert "TIMEZONE=Asia/Shanghai" in content
        assert "MAX_RETRIES=3" in content
        assert "KEYBOARD_LAYOUT=us" in content
        assert "ENC[" not in content  # 没有字段需要加密


class TestSecureEnvLoaderPatterns:
    """验证 SecureEnvLoader.SECRET_KEY_PATTERNS 匹配精度"""

    def test_sensitive_patterns_matched(self):
        patterns = SecureEnvLoader.SECRET_KEY_PATTERNS
        sensitive_vars = [
            "DB_PASSWORD", "API_KEY", "SECRET_TOKEN",
            "ACCESS_KEY", "WEBHOOK_SECRET", "PRIVATE_KEY",
        ]
        for var in sensitive_vars:
            components = set(var.lower().split("_"))
            assert any(k in components for k in patterns), f"{var} 应该匹配敏感模式"

    def test_non_sensitive_not_matched(self):
        patterns = SecureEnvLoader.SECRET_KEY_PATTERNS
        non_sensitive_vars = [
            "KEYBOARD_LAYOUT", "MAX_RETRIES", "TIMEZONE",
            "LOG_LEVEL", "DEBUG_MODE", "PORT",
        ]
        for var in non_sensitive_vars:
            components = set(var.lower().split("_"))
            assert not any(k in components for k in patterns), f"{var} 不应匹配敏感模式"


class TestMainCLI:
    """测试 CLI 入口的各种参数组合"""

    def test_main_decrypt_valid(self, capsys):
        from security.env_encryptor import main

        encrypted = encrypt_value("cli_test_value")
        test_args = ["prog", "--decrypt", encrypted]
        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "cli_test_value" in captured.out
        assert "WARNING" in captured.out

    def test_main_decrypt_invalid(self, capsys):
        from security.env_encryptor import main

        test_args = ["prog", "--decrypt", "not_valid_enc"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_key_positional(self, monkeypatch, capsys):
        from security.env_encryptor import main

        test_args = ["prog", "TEST_KEY"]
        with patch.object(sys, "argv", test_args):
            with patch("getpass.getpass", return_value="my_value"):
                main()

        captured = capsys.readouterr()
        assert "TEST_KEY=ENC[" in captured.out

    def test_main_interactive(self, monkeypatch, capsys):
        from security.env_encryptor import main

        test_args = ["prog"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.input", return_value="MY_PASSWORD"):
                with patch("getpass.getpass", return_value="secret123"):
                    main()

        captured = capsys.readouterr()
        assert "ENC[" in captured.out
        assert "MY_PASSWORD" in captured.out

    def test_main_encrypt_file(self, tmp_path, monkeypatch, capsys):
        from security.env_encryptor import main

        monkeypatch.setattr(SecurityConfig, "IS_PRODUCTION", False)
        monkeypatch.setattr(SecurityConfig, "IS_CI", False)

        input_file = tmp_path / ".env"
        input_file.write_text("DB_PASSWORD=secret\n", encoding="utf-8")
        output_file = tmp_path / ".env.encrypted"

        test_args = ["prog", "--encrypt-file", str(input_file), "--output", str(output_file)]
        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "Encrypted" in captured.out
        assert output_file.exists()
