"""
术语别名映射管理器
提供用户术语 → 标准术语的映射功能，支持热更新、双向查询
"""

import re
from pathlib import Path
from typing import Dict, Optional
import yaml
import logging

from config.path import PROJECT_ROOT

logger = logging.getLogger(__name__)


class AliasManager:
    """术语别名管理器 — 用户术语 → 标准术语"""

    def __init__(self):
        self._aliases: Dict[str, str] = {}
        self._loaded_files: set = set()
        self._file_mtimes: Dict[str, float] = {}

    def load(self, file_path: Path) -> bool:
        """从 YAML 文件加载别名映射

        格式：
          "工资条": "薪资明细"
          "公积金": "住房公积金"

        返回是否加载成功。
        """
        file_key = str(file_path.absolute())

        # 避免重复加载
        if file_key in self._loaded_files:
            return True

        if not file_path.exists():
            logger.warning("别名文件不存在: %s", file_path)
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if isinstance(data, dict):
                for user_term, standard_term in data.items():
                    user_term = str(user_term).strip()
                    standard_term = str(standard_term).strip()
                    if user_term and standard_term:
                        self._aliases[user_term] = standard_term

            self._loaded_files.add(file_key)
            self._file_mtimes[file_key] = file_path.stat().st_mtime
            logger.info("加载别名文件: %s (%d 条)", file_path.name, len(data) if data else 0)
            return True
        except Exception as e:
            logger.error("加载别名文件失败 %s: %s", file_path, e)
            return False

    def _check_auto_reload(self) -> None:
        """检查已加载文件是否被修改，若文件有变更则自动触发重载"""
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
        if not self._aliases:
            return text

        # 按长度降序排列构建正则，确保最长匹配优先
        sorted_terms = sorted(self._aliases.keys(), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(t) for t in sorted_terms))
        return pattern.sub(lambda m: self._aliases[m.group(0)], text)

    def reload(self) -> None:
        """重新加载所有别名"""
        files = list(self._loaded_files)
        self._aliases.clear()
        self._loaded_files.clear()
        self._file_mtimes.clear()
        for file_key in files:
            self.load(Path(file_key))

    @property
    def aliases(self) -> Dict[str, str]:
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
