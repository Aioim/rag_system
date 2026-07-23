"""security 模块测试共享 fixtures"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet


@pytest.fixture
def temp_key_file():
    """创建临时 Fernet 密钥文件，返回路径"""
    key = Fernet.generate_key()
    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as f:
        f.write(key)
    path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def valid_fernet():
    """返回一个有效的 Fernet 实例和对应的密钥"""
    key = Fernet.generate_key()
    return Fernet(key), key


@pytest.fixture
def clean_env():
    """隔离 os.environ 的 fixture"""
    original = os.environ.copy()
    yield
    # 恢复原始环境变量
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture
def temp_env_file(tmp_path):
    """创建临时 .env 文件，返回路径"""
    def _create(content: str) -> Path:
        env_file = tmp_path / ".env"
        env_file.write_text(content, encoding="utf-8")
        return env_file
    return _create
