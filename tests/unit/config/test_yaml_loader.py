"""YamlLoader 缓存行为测试（审查 H3）"""
from config.yaml_loader import YamlLoader


class TestCacheWithMissingEnvFile:
    """审查 H3：{env}.yaml 不存在时缓存不应被永久判失效"""

    def test_missing_env_yaml_does_not_invalidate_cache(self, tmp_path, monkeypatch):
        """dev.yaml 不存在时返回空字典，第二次 load 应命中缓存"""
        loader = YamlLoader(config_dir=tmp_path)
        loader.load_environment("dev")  # 无 dev.yaml → 空配置，写入缓存

        calls: list[str] = []
        orig = loader._load_yaml_with_mtime
        monkeypatch.setattr(
            loader,
            "_load_yaml_with_mtime",
            lambda f: (calls.append(f), orig(f))[1],
        )

        cfg = loader.load_environment("dev")

        assert cfg == {}
        assert calls == [], "命中缓存时不应重新读取 YAML 文件"

    def test_new_env_yaml_invalidates_cache(self, tmp_path):
        """之后新建 dev.yaml 应使缓存失效，新值生效"""
        loader = YamlLoader(config_dir=tmp_path)
        assert loader.load_environment("dev") == {}

        (tmp_path / "dev.yaml").write_text("a: 2", encoding="utf-8")

        assert loader.load_environment("dev")["a"] == 2

    def test_deleted_env_yaml_invalidates_cache(self, tmp_path):
        """删除已缓存的 dev.yaml 应使缓存失效，返回空字典"""
        (tmp_path / "dev.yaml").write_text("a: 2", encoding="utf-8")
        loader = YamlLoader(config_dir=tmp_path)
        assert loader.load_environment("dev")["a"] == 2

        (tmp_path / "dev.yaml").unlink()

        assert loader.load_environment("dev") == {}
