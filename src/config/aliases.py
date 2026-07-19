"""
术语别名映射管理器
提供用户术语 → 标准术语的映射功能，支持热更新、双向查询
"""

import logging
import re
import threading
import time
from pathlib import Path

import yaml

from config.path import PROJECT_ROOT

logger = logging.getLogger(__name__)


class AliasManager:
    """术语别名管理器 — 用户术语 → 标准术语

    并发约定：写路径（load/reload）持锁，构建完成后整体替换字典引用；
    读路径（resolve/resolve_all）取引用快照免锁，不会读到半重建状态。
    """

    # 自动重载的文件 mtime 检查间隔（秒）——不必每次 resolve 都 stat
    _MTIME_CHECK_INTERVAL_SECONDS: float = 2.0

    def __init__(self):
        self._aliases: dict[str, str] = {}
        self._loaded_files: set = set()
        self._file_mtimes: dict[str, float] = {}
        self._lock = threading.RLock()
        # resolve_all 正则缓存，_aliases 变更时随 _cache_key 一起失效
        self._cache_key: int = 0
        self._cache_version: int = 0
        self._cached_pattern: re.Pattern | None = None
        self._cached_sorted_terms: tuple[str, ...] = ()
        self._last_mtime_check: float = 0.0

    @staticmethod
    def _parse_file(file_path: Path) -> tuple[dict[str, str], float] | None:
        """解析单个别名文件，返回 (映射, mtime)；文件缺失或解析失败返回 None

        格式：
          "工资条": "薪资明细"
          "公积金": "住房公积金"
        """
        if not file_path.exists():
            logger.warning("别名文件不存在: %s", file_path)
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            mapping: dict[str, str] = {}
            if isinstance(data, dict):
                for user_term, standard_term in data.items():
                    user_term = str(user_term).strip()
                    standard_term = str(standard_term).strip()
                    if user_term and standard_term:
                        mapping[user_term] = standard_term
            return mapping, file_path.stat().st_mtime
        except Exception as e:
            logger.error("加载别名文件失败 %s: %s", file_path, e)
            return None

    def load(self, file_path: Path) -> bool:
        """从 YAML 文件加载别名映射，返回是否加载成功。"""
        file_key = str(file_path.absolute())

        with self._lock:
            # 避免重复加载
            if file_key in self._loaded_files:
                return True

            parsed = self._parse_file(file_path)
            if parsed is None:
                return False

            mapping, mtime = parsed
            # 整体替换引用（不原地修改），保证读路径快照一致
            self._aliases = {**self._aliases, **mapping}
            self._loaded_files = self._loaded_files | {file_key}
            self._file_mtimes = {**self._file_mtimes, file_key: mtime}
            logger.info("加载别名文件: %s (%d 条)", file_path.name, len(mapping))
            return True

    def _check_auto_reload(self) -> None:
        """检查已加载文件是否被修改（间隔节流，避免每次 resolve 都 stat）

        若 settings.aliases.auto_reload 未就绪默认视为开启。
        """
        _now = time.monotonic()
        if _now - self._last_mtime_check < self._MTIME_CHECK_INTERVAL_SECONDS:
            return
        self._last_mtime_check = _now

        needs_reload = False
        for file_key, cached_mtime in list(self._file_mtimes.items()):
            file_path = Path(file_key)
            if not file_path.exists():
                needs_reload = True
                break
            try:
                if file_path.stat().st_mtime > cached_mtime:
                    needs_reload = True
                    break
            except OSError:
                continue
        if needs_reload:
            logger.info("检测到别名文件变更，自动重载")
            self.reload()

    def resolve(self, user_term: str) -> str:
        """将用户术语映射为标准术语，未找到返回原词（自动检测文件变更）"""
        self._check_auto_reload()
        return self._aliases.get(user_term, user_term)

    def resolve_all(self, text: str) -> str:
        """将文本中所有已知的用户术语替换为标准术语（最长匹配优先，单次扫描，自动检测文件变更）"""
        self._check_auto_reload()
        aliases = self._aliases  # 引用快照：构建正则与查值使用同一份映射
        if not aliases:
            return text

        # —— 正则缓存（_aliases 版本号不变且缓存就绪时可复用） ——
        current_version = self._cache_version
        if current_version != self._cache_key or self._cached_pattern is None:
            sorted_terms = tuple(sorted(aliases.keys(), key=len, reverse=True))
            self._cached_sorted_terms = sorted_terms
            self._cached_pattern = re.compile(
                "|".join(re.escape(t) for t in sorted_terms)
            )
            self._cache_key = current_version
        return self._cached_pattern.sub(
            lambda m: aliases[m.group(0)], text
        )

    def reload(self) -> None:
        """重新加载所有别名（局部构建完成后一次性发布，读路径不会见到中间态）

        临时解析失败的文件保留上一次成功的映射，避免磁盘抖动导致别名静默丢失。
        """
        with self._lock:
            new_aliases: dict[str, str] = dict(self._aliases)
            new_loaded: set = set()
            new_mtimes: dict[str, float] = {}
            for file_key in self._loaded_files:
                parsed = self._parse_file(Path(file_key))
                if parsed is None:
                    # 保留上一次成功的映射（文件可能临时无法访问）
                    new_loaded.add(file_key)
                    new_mtimes[file_key] = self._file_mtimes.get(file_key, 0.0)
                    continue
                mapping, mtime = parsed
                new_aliases.update(mapping)
                new_loaded.add(file_key)
                new_mtimes[file_key] = mtime

            self._aliases = new_aliases
            self._loaded_files = new_loaded
            self._file_mtimes = new_mtimes
            # 正则缓存随别名变更失效（先置缓存键为无效值，再清空，避免读路径见到匹配但为 None 的缓存）
            self._cache_key = -1
            self._cached_pattern = None
            self._cache_version += 1
            self._cache_key = self._cache_version

    @property
    def aliases(self) -> dict[str, str]:
        """返回所有别名映射的只读副本"""
        return dict(self._aliases)

    @property
    def count(self) -> int:
        return len(self._aliases)


# ============================================================================
# 全局单例
# ============================================================================

alias_manager = AliasManager()

# 启动时尝试从默认路径加载（配置文件路径由 ConfigManager.initialize 后续再加载）
_default_alias_path = PROJECT_ROOT / "config" / "aliases.yaml"
if _default_alias_path.exists():
    alias_manager.load(_default_alias_path)


def resolve_alias(user_term: str) -> str:
    """便捷函数：将用户术语映射为标准术语"""
    return alias_manager.resolve(user_term)


def resolve_aliases_in_text(text: str) -> str:
    """便捷函数：替换文本中所有已知别名"""
    return alias_manager.resolve_all(text)
