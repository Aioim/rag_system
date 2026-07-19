"""settings 配置模型测试（审查 H1/H2/H5）"""
import threading

import pytest

from config.path import PROJECT_ROOT
from config.settings import (
    AliasConfig,
    FaissConfig,
    FinetuneConfig,
    IngestionConfig,
    LogConfig,
    ModelConfig,
    RAGAppConfig,
    SessionConfig,
    settings,
)


class TestPathNormalization:
    """审查 H1：YAML 相对路径必须归一化到 PROJECT_ROOT，不依赖 CWD"""

    @pytest.mark.parametrize(
        "cfg_cls,field,rel",
        [
            (SessionConfig, "db_path", "data/sessions.db"),
            (LogConfig, "log_dir", "logs"),
            (FaissConfig, "index_dir", "data/faiss_indexes"),
            (AliasConfig, "file_path", "config/aliases.yaml"),
            (ModelConfig, "cache_dir", "models"),
            (IngestionConfig, "parsed_doc_dir", "data/parsed_docs"),
            (FinetuneConfig, "data_dir", "data/finetune"),
            (FinetuneConfig, "output_dir", "models/finetuned"),
        ],
    )
    def test_relative_path_resolved_to_project_root(self, cfg_cls, field, rel):
        cfg = cfg_cls(**{field: rel})

        value = getattr(cfg, field)

        assert value.is_absolute(), f"{cfg_cls.__name__}.{field} 应为绝对路径"
        assert value == PROJECT_ROOT / rel

    def test_absolute_path_unchanged(self):
        abs_path = PROJECT_ROOT / "custom" / "x.db"

        cfg = SessionConfig(db_path=str(abs_path))

        assert cfg.db_path == abs_path


class TestConfigManagerGetThreadSafety:
    """审查 H2：get() 的 _dict_cache 惰性构建与 reload() 并发时不应出错"""

    def test_get_returns_value_after_reload(self):
        before = settings.get("retrieval.top_k")
        settings.reload()

        after = settings.get("retrieval.top_k")

        assert after == before
        assert after is not None

    def test_concurrent_get_and_reload_no_error(self):
        errors: list = []

        def reader():
            try:
                for _ in range(300):
                    if settings.get("retrieval.top_k") is None:
                        errors.append("get 返回了 None")
            except Exception as e:  # noqa: BLE001 — 测试收集任意异常
                errors.append(repr(e))

        def reloader():
            try:
                for _ in range(3):
                    settings.reload()
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=reloader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


class TestFromEnvWhitelist:
    """审查 H5：环境变量注入应按配置段白名单过滤，而非系统变量黑名单"""

    def test_unknown_user_vars_not_injected(self, monkeypatch):
        """与配置无关的用户变量（DB_HOST/MY_TOOL_OPT 等）不应进入配置"""
        monkeypatch.setenv("DB_HOST", "10.0.0.1")
        monkeypatch.setenv("MY_TOOL_OPT", "x")

        data = RAGAppConfig.from_env()

        assert "db_host" not in data
        assert "my_tool_opt" not in data

    def test_unknown_nested_vars_not_injected(self, monkeypatch):
        """非配置段根名的嵌套变量不应进入配置"""
        monkeypatch.setenv("FOO__BAR", "1")

        data = RAGAppConfig.from_env()

        assert "foo" not in data

    def test_documented_nested_override_works(self, monkeypatch):
        """文档形态 RETRIEVAL__TOP_K=10 应正常注入"""
        monkeypatch.setenv("RETRIEVAL__TOP_K", "10")

        data = RAGAppConfig.from_env()

        assert data["retrieval"]["top_k"] == 10

    def test_env_and_debug_scalars_injected(self, monkeypatch):
        """顶层标量 ENV/DEBUG 应正常注入"""
        monkeypatch.setenv("DEBUG", "true")

        data = RAGAppConfig.from_env()

        assert data["debug"] is True

    def test_rag_prefix_escape_hatch(self, monkeypatch):
        """RAG__ 前缀应绕过白名单注入任意键"""
        monkeypatch.setenv("RAG__MY_KEY", "val")

        data = RAGAppConfig.from_env()

        assert data["my_key"] == "val"

    def test_system_vars_not_injected(self):
        """真实系统变量（PATH/TEMP 等，无 __）不应进入配置"""
        data = RAGAppConfig.from_env()

        assert "path" not in data
        assert "temp" not in data
        assert "os" not in data
