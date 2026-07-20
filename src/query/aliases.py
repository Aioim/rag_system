"""
术语别名映射管理器 — 支持多别名、多含义上下文消歧

提供用户术语 → 标准术语的映射功能，支持：
- 多别名：一个标准术语对应多个用户口语表达
- 多含义：同一用户术语在不同语境下映射到不同标准术语（通过 context 关键词消歧）
- 热重载：自动检测 YAML 文件变更
- 并发安全：写路径持锁构建，读路径引用快照免锁

YAML 格式（新旧兼容）：

    旧格式（扁平 key-value）：
      "工资条": "薪资明细"

    新格式（结构化条目）：
      entries:
        - aliases: ["工资条", "工资单"]
          target: "薪资明细"
        - aliases: ["系统"]
          target: "内部系统"
          context: ["IT", "登录", "vpn"]
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from config.path import PROJECT_ROOT

logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型
# ============================================================================


@dataclass(frozen=True)
class AliasEntry:
    """单个别名条目 — 一组用户术语 → 一个标准术语（可选消歧上下文）"""

    aliases: tuple[str, ...]
    """用户可能使用的术语列表（如 ["工资条", "工资单"]）"""

    target: str
    """标准术语（如 "薪资明细"）"""

    context: tuple[str, ...] = ()
    """消歧关键词列表；当同一别名有多个条目时用于上下文匹配"""


# ============================================================================
# 别名管理器
# ============================================================================


class AliasManager:
    """术语别名管理器 — 多别名 + 多含义消歧

    并发约定：写路径（load/reload）持锁，构建完成后整体替换字典引用；
    读路径（resolve/resolve_all/get_candidates）取引用快照免锁。
    """

    _MTIME_CHECK_INTERVAL_SECONDS: float = 2.0

    def __init__(self) -> None:
        # alias → 候选条目列表（同一别名可能有多个条目，对应不同含义）
        self._aliases: dict[str, list[AliasEntry]] = {}
        self._loaded_files: set[str] = set()
        self._file_mtimes: dict[str, float] = {}
        self._file_alias_keys: dict[str, set[str]] = {}  # 每个文件贡献了哪些别名
        self._lock = threading.RLock()

        # resolve_all 正则缓存
        self._cache_key: int = 0
        self._cache_version: int = 0
        self._cached_pattern: re.Pattern | None = None
        self._last_mtime_check: float = 0.0

    # ------------------------------------------------------------------
    # YAML 解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_file(
        file_path: Path,
    ) -> tuple[dict[str, list[AliasEntry]], float] | None:
        """解析单个别名文件，返回 (索引, mtime)；文件缺失或解析失败返回 None

        兼容两种 YAML 格式：
        - 旧格式：扁平 dict[str, str]
        - 新格式：``entries: [{aliases: [...], target: ..., context?: [...]}, ...]``
        """
        if not file_path.exists():
            logger.warning("别名文件不存在: %s", file_path)
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error("加载别名文件失败 %s: %s", file_path, e)
            return None

        raw_entries: list[dict] = []

        if isinstance(data, dict):
            if "entries" in data:
                # —— 新格式 ——
                entries_list = data["entries"]
                if isinstance(entries_list, list):
                    raw_entries = entries_list
            else:
                # —— 旧格式：扁平 key-value ——
                for user_term, standard_term in data.items():
                    if user_term and standard_term:
                        raw_entries.append({
                            "aliases": [str(user_term).strip()],
                            "target": str(standard_term).strip(),
                        })

        # 解析为 AliasEntry 列表
        parsed: list[AliasEntry] = []
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            aliases_raw = item.get("aliases", [])
            if isinstance(aliases_raw, str):
                aliases_raw = [aliases_raw]
            aliases = tuple(str(a).strip() for a in aliases_raw if a)
            target = str(item.get("target", "")).strip()
            if not aliases or not target:
                continue
            context_raw = item.get("context", [])
            if isinstance(context_raw, str):
                context_raw = [context_raw]
            context = tuple(str(c).strip() for c in context_raw if c)
            parsed.append(AliasEntry(aliases=aliases, target=target, context=context))

        # 构建 alias → [AliasEntry] 索引
        index: dict[str, list[AliasEntry]] = {}
        for entry in parsed:
            for alias in entry.aliases:
                index.setdefault(alias, []).append(entry)

        return index, file_path.stat().st_mtime

    # ------------------------------------------------------------------
    # 加载与重载
    # ------------------------------------------------------------------

    def load(self, file_path: Path) -> bool:
        """从 YAML 文件加载别名映射，返回是否加载成功。"""
        file_key = str(file_path.absolute())

        with self._lock:
            if file_key in self._loaded_files:
                return True

            parsed = self._parse_file(file_path)
            if parsed is None:
                return False

            index, mtime = parsed
            # 整体替换引用（不原地修改），保证读路径快照一致
            new_aliases: dict[str, list[AliasEntry]] = dict(self._aliases)
            for alias, entries in index.items():
                existing = new_aliases.setdefault(alias, [])
                # 避免完全重复的条目（同一 alias+target+context 组合）
                for entry in entries:
                    if entry not in existing:
                        existing.append(entry)

            self._aliases = new_aliases
            self._loaded_files = self._loaded_files | {file_key}
            self._file_mtimes = {**self._file_mtimes, file_key: mtime}
            self._file_alias_keys = {**self._file_alias_keys, file_key: set(index.keys())}
            self._invalidate_cache()
            logger.info("加载别名文件: %s (%d 条)", file_path.name, len(index))
            return True

    def reload(self) -> None:
        """重新加载所有别名（局部构建完成后一次性发布，读路径不会见到中间态）

        临时解析失败的文件仅保留该文件上一次成功的映射，不影响其他文件的更新。
        """
        with self._lock:
            new_aliases: dict[str, list[AliasEntry]] = {}
            new_loaded: set[str] = set()
            new_mtimes: dict[str, float] = {}
            new_file_alias_keys: dict[str, set[str]] = {}

            for file_key in self._loaded_files:
                parsed = self._parse_file(Path(file_key))
                if parsed is None:
                    # 仅恢复该文件上一次成功加载的别名（而非全部文件）
                    new_loaded.add(file_key)
                    new_mtimes[file_key] = self._file_mtimes.get(file_key, 0.0)
                    old_keys = self._file_alias_keys.get(file_key, set())
                    new_file_alias_keys[file_key] = old_keys
                    for alias in old_keys:
                        old_entries = self._aliases.get(alias, [])
                        for entry in old_entries:
                            existing = new_aliases.setdefault(alias, [])
                            if entry not in existing:
                                existing.append(entry)
                    continue
                index, mtime = parsed
                for alias, entries in index.items():
                    existing = new_aliases.setdefault(alias, [])
                    for entry in entries:
                        if entry not in existing:
                            existing.append(entry)
                new_loaded.add(file_key)
                new_mtimes[file_key] = mtime
                new_file_alias_keys[file_key] = set(index.keys())

            self._aliases = new_aliases
            self._loaded_files = new_loaded
            self._file_mtimes = new_mtimes
            self._file_alias_keys = new_file_alias_keys
            self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """使 resolve_all 正则缓存失效"""
        self._cached_pattern = None
        self._cache_version += 1
        self._cache_key = self._cache_version

    # ------------------------------------------------------------------
    # 消歧
    # ------------------------------------------------------------------

    @staticmethod
    def _disambiguate(
        candidates: list[AliasEntry], context_text: str
    ) -> str | None:
        """从多个候选中选出最匹配上下文的 target，无可匹配时返回 None

        评分：context 关键词在 context_text 中的命中次数。
        相同时返回第一个最高分（确定性）。
        """
        best_target: str | None = None
        best_score = 0
        for entry in candidates:
            if not entry.context:
                continue
            score = sum(1 for kw in entry.context if kw in context_text)
            if score > best_score:
                best_score = score
                best_target = entry.target
        return best_target

    # ------------------------------------------------------------------
    # 查询 API
    # ------------------------------------------------------------------

    def _check_auto_reload(self) -> None:
        """检查已加载文件是否被修改（间隔节流）"""
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

    def resolve(self, user_term: str, context_text: str = "") -> str:
        """将用户术语映射为标准术语

        Args:
            user_term: 用户输入的术语
            context_text: 消歧上下文（如完整查询文本）；歧义时用于匹配 context 关键词

        Returns:
            标准术语；若未找到或歧义无法消解则返回原词
        """
        self._check_auto_reload()
        candidates = self._aliases.get(user_term)
        if not candidates:
            return user_term
        if len(candidates) == 1:
            return candidates[0].target
        if context_text:
            best = self._disambiguate(candidates, context_text)
            if best is not None:
                return best
        return user_term

    def resolve_all(self, text: str) -> str:
        """将文本中所有已知的用户术语替换为标准术语

        对每个匹配的术语：
        - 单候选 → 直接替换
        - 多候选 → 用全文做消歧上下文，匹配 context 关键词
        - 歧义无法消解 → 保留原词

        最长匹配优先，单次扫描。
        """
        self._check_auto_reload()
        aliases = self._aliases  # 引用快照
        if not aliases:
            return text

        # —— 正则缓存 ——
        pattern = self._cached_pattern
        current_version = self._cache_version
        if current_version != self._cache_key or pattern is None:
            sorted_terms = sorted(aliases.keys(), key=len, reverse=True)
            pattern = re.compile(
                "|".join(re.escape(t) for t in sorted_terms)
            )
            self._cached_pattern = pattern
            self._cache_key = current_version

        # —— 预计算每个别名的替换目标（全文消歧） ——
        replacement_map: dict[str, str] = {}
        for term, candidates in aliases.items():
            if len(candidates) == 1:
                replacement_map[term] = candidates[0].target
            else:
                best = self._disambiguate(candidates, text)
                replacement_map[term] = best if best is not None else term

        return pattern.sub(
            lambda m: replacement_map[m.group(0)], text
        )

    def get_candidates(self, user_term: str) -> list[str]:
        """返回某用户术语的所有可能标准术语（用于下游自行消歧）"""
        self._check_auto_reload()
        candidates = self._aliases.get(user_term)
        if not candidates:
            return []
        return [e.target for e in candidates]

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def aliases(self) -> dict[str, list[str]]:
        """返回所有别名映射的只读副本（alias → 可能的 target 列表）"""
        return {
            alias: [e.target for e in entries]
            for alias, entries in self._aliases.items()
        }

    @property
    def count(self) -> int:
        """返回唯一别名数量（非条目数）"""
        return len(self._aliases)


# ============================================================================
# 全局单例
# ============================================================================

alias_manager = AliasManager()

# 启动时尝试从默认路径加载
_default_alias_path = PROJECT_ROOT / "config" / "aliases.yaml"
if _default_alias_path.exists():
    alias_manager.load(_default_alias_path)


def resolve_alias(user_term: str, context_text: str = "") -> str:
    """便捷函数：将用户术语映射为标准术语

    Args:
        user_term: 用户术语
        context_text: 可选的消歧上下文；歧义时用于匹配 context 关键词
    """
    return alias_manager.resolve(user_term, context_text)


def resolve_aliases_in_text(text: str) -> str:
    """便捷函数：替换文本中所有已知别名，自动全文消歧"""
    return alias_manager.resolve_all(text)
