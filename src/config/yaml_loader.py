"""YAML 配置加载器 — 支持多环境覆盖和缓存"""

from pathlib import Path
from typing import Any, Dict, Tuple
import yaml
from config.path import PROJECT_ROOT


def deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典，override 中的值覆盖 base（被 YamlLoader 和 RAGAppConfig 共用）"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class YamlLoader:
    """YAML 配置加载器（支持缓存和文件修改时间检测）"""

    def __init__(self, config_dir: Path | None = None):
        if config_dir is None:
            config_dir = PROJECT_ROOT / "config"
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Tuple[Dict[str, Any], Dict[str, float]]] = {}

    def load_environment(self, env: str = "dev") -> Dict[str, Any]:
        """
        加载指定环境的 YAML 配置。
        合并策略：defaults.yaml → {env}.yaml（后者覆盖前者）
        """
        # 检查缓存
        if env in self._cache:
            cached_config, mtime_dict = self._cache[env]
            if self._is_cache_valid(mtime_dict):
                return deep_merge({}, cached_config)  # 浅拷贝代替 deepcopy

        # 加载 defaults.yaml 和 {env}.yaml
        base_config, base_mtime = self._load_yaml_with_mtime("defaults.yaml")
        env_file = f"{env}.yaml"
        env_config, env_mtime = (
            self._load_yaml_with_mtime(env_file)
            if (self.config_dir / env_file).exists()
            else ({}, 0)
        )

        merged = deep_merge(base_config, env_config)
        mtime_dict = {"defaults.yaml": base_mtime, f"{env}.yaml": env_mtime}
        self._cache[env] = (merged, mtime_dict)
        return deep_merge({}, merged)  # 返回副本防止外部修改缓存

    def _is_cache_valid(self, mtime_dict: Dict[str, float]) -> bool:
        for filename, cached_mtime in mtime_dict.items():
            file_path = self.config_dir / filename
            if not file_path.exists():
                return False  # 文件被删除，缓存失效
            if file_path.stat().st_mtime > cached_mtime:
                return False  # 文件已更新，缓存失效
        return True

    def _load_yaml_with_mtime(self, filename: str) -> Tuple[Dict[str, Any], float]:
        file_path = self.config_dir / filename
        if not file_path.exists():
            if filename == "defaults.yaml":
                # 默认配置文件不存在时返回空配置（不抛异常，依赖代码默认值）
                return {}, 0.0
            return {}, 0.0

        mtime = file_path.stat().st_mtime
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config, mtime

    def clear_cache(self) -> None:
        self._cache.clear()


if __name__ == "__main__":
    loader = YamlLoader()
    config = loader.load_environment()
    print("配置加载成功，顶层键:", list(config.keys()))
