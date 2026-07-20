"""YAML 配置加载器 — 支持多环境覆盖和缓存"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from config.path import PROJECT_ROOT


def deep_merge(base: dict, override: dict) -> dict:
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
        self._cache: dict[str, tuple[dict[str, Any], dict[str, float]]] = {}

    def load_environment(self, env: str = "dev") -> dict[str, Any]:
        """
        加载指定环境的 YAML 配置。
        直接读取 {env}.yaml，不再合并 defaults.yaml。
        若 {env}.yaml 不存在则返回空字典（依赖 Pydantic 模型默认值）。
        """
        env_file = f"{env}.yaml"

        # 检查缓存
        if env in self._cache:
            cached_config, mtime_dict = self._cache[env]
            if self._is_cache_valid(mtime_dict):
                return deepcopy(cached_config)  # 深拷贝防止外部修改缓存

        # 直接加载 {env}.yaml
        env_config, env_mtime = self._load_yaml_with_mtime(env_file)
        mtime_dict = {env_file: env_mtime}
        self._cache[env] = (env_config, mtime_dict)
        return deepcopy(env_config)  # 深拷贝防止外部修改缓存

    def _is_cache_valid(self, mtime_dict: dict[str, float]) -> bool:
        for filename, cached_mtime in mtime_dict.items():
            file_path = self.config_dir / filename
            if not file_path.exists():
                if cached_mtime:
                    return False  # 文件被删除，缓存失效
                continue  # 缓存时就不存在（mtime=0），仍不存在视为未变化
            if file_path.stat().st_mtime > cached_mtime:
                return False  # 文件已更新（或新创建），缓存失效
        return True

    def _load_yaml_with_mtime(self, filename: str) -> tuple[dict[str, Any], float]:
        file_path = self.config_dir / filename
        if not file_path.exists():
            return {}, 0.0

        mtime = file_path.stat().st_mtime
        with open(file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config, mtime

    def clear_cache(self) -> None:
        self._cache.clear()


if __name__ == "__main__":
    loader = YamlLoader()
    config = loader.load_environment()
    print("配置加载成功，顶层键:", list(config.keys()))
