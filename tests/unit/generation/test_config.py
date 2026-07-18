"""GenerationConfig 配置模型测试"""
import pytest
from pydantic import ValidationError

from config import settings
from config.settings import GenerationConfig, RAGAppConfig


class TestDefaults:
    def test_dedup_threshold_default(self):
        assert settings.generation.dedup_threshold == 0.85

    def test_max_context_chars_default(self):
        assert settings.generation.max_context_chars == 9000

    def test_fact_check_enabled_default(self):
        assert settings.generation.fact_check_enabled is True

    def test_max_query_chars_default(self):
        assert settings.generation.max_query_chars == 2000


class TestFieldConstraints:
    """非法配置在加载时被拒绝，而非运行时静默出错"""

    def test_dedup_threshold_rejects_out_of_range(self):
        with pytest.raises(ValidationError):
            GenerationConfig(dedup_threshold=1.5)
        with pytest.raises(ValidationError):
            GenerationConfig(dedup_threshold=-0.1)

    def test_dedup_threshold_accepts_boundaries(self):
        assert GenerationConfig(dedup_threshold=0.0).dedup_threshold == 0.0
        assert GenerationConfig(dedup_threshold=1.0).dedup_threshold == 1.0

    def test_max_context_chars_rejects_non_positive(self):
        with pytest.raises(ValidationError):
            GenerationConfig(max_context_chars=0)
        with pytest.raises(ValidationError):
            GenerationConfig(max_context_chars=-100)


class TestEnvOverride:
    """环境变量 GENERATION__* 覆盖生效（不触碰全局单例）"""

    def test_env_override_parsed(self, monkeypatch):
        monkeypatch.setenv("GENERATION__DEDUP_THRESHOLD", "0.9")
        env_data = RAGAppConfig.from_env()
        assert env_data["generation"]["dedup_threshold"] == "0.9"

    def test_env_override_coerced_by_pydantic(self):
        cfg = RAGAppConfig(generation={"dedup_threshold": "0.9"})
        assert cfg.generation.dedup_threshold == 0.9
