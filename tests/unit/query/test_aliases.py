"""AliasManager 测试 — 多别名、多含义消歧、热重载、并发安全"""

import threading
import time

import pytest
import yaml as yaml_mod

from query.aliases import AliasEntry, AliasManager


# ============================================================================
# YAML 格式兼容性
# ============================================================================


class TestYamlFormat:
    def test_old_flat_format(self, tmp_path):
        """旧格式（扁平 key-value）应正确解析为单 alias 条目"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text('"工资条": "薪资明细"\n"公积金": "住房公积金"\n', encoding="utf-8")

        mgr = AliasManager()
        assert mgr.load(alias_file)
        assert mgr.resolve("工资条") == "薪资明细"
        assert mgr.resolve("公积金") == "住房公积金"
        assert mgr.count == 2

    def test_new_entries_format(self, tmp_path):
        """新格式（entries 列表）应正确解析多别名条目"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text(
            "entries:\n"
            '  - aliases: ["工资条", "工资单"]\n'
            '    target: "薪资明细"\n'
            '  - aliases: ["公积金"]\n'
            '    target: "住房公积金"\n',
            encoding="utf-8",
        )

        mgr = AliasManager()
        assert mgr.load(alias_file)
        assert mgr.resolve("工资条") == "薪资明细"
        assert mgr.resolve("工资单") == "薪资明细"
        assert mgr.resolve("公积金") == "住房公积金"

    def test_mixed_format_loads_both(self, tmp_path):
        """先后加载新旧两种格式应正确合并"""
        old_file = tmp_path / "old.yaml"
        new_file = tmp_path / "new.yaml"
        old_file.write_text('"个税": "个人所得税"\n', encoding="utf-8")
        new_file.write_text(
            "entries:\n"
            '  - aliases: ["工资条", "工资单"]\n'
            '    target: "薪资明细"\n',
            encoding="utf-8",
        )

        mgr = AliasManager()
        assert mgr.load(old_file)
        assert mgr.load(new_file)
        assert mgr.resolve("个税") == "个人所得税"
        assert mgr.resolve("工资条") == "薪资明细"
        assert mgr.resolve("工资单") == "薪资明细"


# ============================================================================
# 多别名
# ============================================================================


class TestMultiAlias:
    def test_multiple_aliases_same_target(self, tmp_path):
        """多个别名都指向同一个标准术语"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text(
            "entries:\n"
            '  - aliases: ["工资条", "工资单", "薪资单"]\n'
            '    target: "薪资明细"\n',
            encoding="utf-8",
        )

        mgr = AliasManager()
        mgr.load(alias_file)
        assert mgr.resolve("工资条") == "薪资明细"
        assert mgr.resolve("工资单") == "薪资明细"
        assert mgr.resolve("薪资单") == "薪资明细"
        assert mgr.count == 3  # 3 个别名

    def test_resolve_all_replaces_all_aliases(self, tmp_path):
        """resolve_all 应替换文本中所有已知别名"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text(
            "entries:\n"
            '  - aliases: ["工资条", "工资单"]\n'
            '    target: "薪资明细"\n'
            '  - aliases: ["公积金"]\n'
            '    target: "住房公积金"\n',
            encoding="utf-8",
        )

        mgr = AliasManager()
        mgr.load(alias_file)
        result = mgr.resolve_all("我的工资单和公积金怎么查")
        assert result == "我的薪资明细和住房公积金怎么查"


# ============================================================================
# 多含义消歧
# ============================================================================


class TestDisambiguation:
    @pytest.fixture
    def mgr_with_polysemy(self, tmp_path):
        """创建一个包含多义词的 AliasManager"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text(
            "entries:\n"
            '  - aliases: ["系统"]\n'
            '    target: "内部系统"\n'
            "    context: [IT, 登录, 账号, 权限, vpn, wifi, 打印机, 电脑]\n"
            '  - aliases: ["系统"]\n'
            '    target: "考勤系统"\n'
            "    context: [打卡, 考勤, 请假, 休假, 迟到, 早退]\n"
            '  - aliases: ["工资条"]\n'
            '    target: "薪资明细"\n',
            encoding="utf-8",
        )
        mgr = AliasManager()
        mgr.load(alias_file)
        return mgr

    def test_disambiguate_by_context(self, mgr_with_polysemy):
        """根据上下文消歧 — "系统" 在不同语境下映射到不同 target"""
        mgr = mgr_with_polysemy
        # IT 语境
        assert mgr.resolve("系统", context_text="IT系统登录不了") == "内部系统"
        assert mgr.resolve("系统", context_text="vpn连不上") == "内部系统"
        # HR 语境
        assert mgr.resolve("系统", context_text="考勤系统怎么打卡") == "考勤系统"
        assert mgr.resolve("系统", context_text="请假流程") == "考勤系统"

    def test_ambiguous_without_context_keeps_original(self, mgr_with_polysemy):
        """无上下文时歧义别名保留原词"""
        mgr = mgr_with_polysemy
        assert mgr.resolve("系统") == "系统"

    def test_ambiguous_no_context_match_keeps_original(self, mgr_with_polysemy):
        """上下文不匹配任何关键词时保留原词"""
        mgr = mgr_with_polysemy
        assert mgr.resolve("系统", context_text="今天天气不错") == "系统"

    def test_resolve_all_with_disambiguation(self, mgr_with_polysemy):
        """resolve_all 应使用全文作为消歧上下文"""
        mgr = mgr_with_polysemy
        # 全文偏向 IT 语境
        result = mgr.resolve_all("IT系统登录有问题，工资条也查不到")
        assert "内部系统" in result
        assert "薪资明细" in result
        # 全文偏向 HR 语境
        result = mgr.resolve_all("考勤系统打不了卡")
        assert "考勤系统" in result

    def test_resolve_all_ambiguous_preserves_original(self, mgr_with_polysemy):
        """resolve_all 歧义无法消解时保留原词"""
        mgr = mgr_with_polysemy
        result = mgr.resolve_all("系统好像有问题")
        # "系统" 在无上下文关键词时保留原词
        assert "系统好像有问题" == result

    def test_unambiguous_alias_always_resolves(self, mgr_with_polysemy):
        """无歧义别名无论在什么上下文中都应正确解析"""
        mgr = mgr_with_polysemy
        assert mgr.resolve("工资条") == "薪资明细"
        assert mgr.resolve("工资条", context_text="随便什么") == "薪资明细"

    def test_get_candidates_returns_all_targets(self, mgr_with_polysemy):
        """get_candidates 应返回所有可能的 target"""
        mgr = mgr_with_polysemy
        candidates = mgr.get_candidates("系统")
        assert sorted(candidates) == ["内部系统", "考勤系统"]

        candidates = mgr.get_candidates("工资条")
        assert candidates == ["薪资明细"]

        candidates = mgr.get_candidates("不存在")
        assert candidates == []


# ============================================================================
# 热重载
# ============================================================================


class TestAliasManagerReload:
    def test_reload_picks_up_file_changes(self, tmp_path):
        """reload 后应读到文件的最新内容"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text(
            "entries:\n"
            '  - aliases: ["工资条"]\n'
            '    target: "薪资明细"\n',
            encoding="utf-8",
        )

        mgr = AliasManager()
        assert mgr.load(alias_file)
        assert mgr.resolve("工资条") == "薪资明细"

        # 修改文件
        alias_file.write_text(
            "entries:\n"
            '  - aliases: ["工资条"]\n'
            '    target: "工资单"\n',
            encoding="utf-8",
        )
        mgr.reload()

        assert mgr.resolve("工资条") == "工资单"
        assert mgr.count == 1

    def test_concurrent_resolve_during_reload_never_sees_partial_state(
        self, tmp_path, monkeypatch
    ):
        """reload 期间并发 resolve/resolve_all 不得读到半清空状态"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text(
            "entries:\n"
            '  - aliases: ["工资条"]\n'
            '    target: "薪资明细"\n',
            encoding="utf-8",
        )

        mgr = AliasManager()
        assert mgr.load(alias_file)

        real_safe_load = yaml_mod.safe_load

        def slow_safe_load(stream):
            time.sleep(0.05)
            return real_safe_load(stream)

        monkeypatch.setattr(yaml_mod, "safe_load", slow_safe_load)

        errors: list[str] = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    if mgr.resolve("工资条") != "薪资明细":
                        errors.append("resolve 读到未完成的重载状态")
                        return
                    if mgr.resolve_all("我的工资条在哪") != "我的薪资明细在哪":
                        errors.append("resolve_all 读到未完成的重载状态")
                        return
                except Exception as e:
                    errors.append(f"并发读取抛出异常: {e!r}")
                    return

        t = threading.Thread(target=reader)
        t.start()
        try:
            for _ in range(10):
                mgr.reload()
        finally:
            stop.set()
            t.join(timeout=10)

        assert not errors, errors[0]
        assert mgr.resolve("工资条") == "薪资明细"


# ============================================================================
# 边界条件
# ============================================================================


class TestEdgeCases:
    def test_empty_file(self, tmp_path):
        """空文件应优雅处理"""
        alias_file = tmp_path / "empty.yaml"
        alias_file.write_text("", encoding="utf-8")
        mgr = AliasManager()
        assert mgr.load(alias_file)
        assert mgr.count == 0

    def test_missing_file(self, tmp_path):
        """不存在的文件 load 返回 False"""
        mgr = AliasManager()
        assert not mgr.load(tmp_path / "nonexistent.yaml")

    def test_resolve_unknown_term(self):
        """未知术语返回原词"""
        mgr = AliasManager()
        assert mgr.resolve("不存在的术语") == "不存在的术语"

    def test_resolve_all_no_aliases(self):
        """无别名时 resolve_all 原样返回"""
        mgr = AliasManager()
        assert mgr.resolve_all("任意文本") == "任意文本"

    def test_resolve_all_longest_match_first(self, tmp_path):
        """最长匹配优先 — "加班餐补" 不应被 "加班" 部分匹配"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text(
            "entries:\n"
            '  - aliases: ["加班餐补"]\n'
            '    target: "加班餐费补贴"\n'
            '  - aliases: ["加班"]\n'
            '    target: "加班管理"\n',
            encoding="utf-8",
        )
        mgr = AliasManager()
        mgr.load(alias_file)
        # "加班餐补" 是更长的匹配，应优先
        result = mgr.resolve_all("加班餐补申请")
        assert result == "加班餐费补贴申请"


# ============================================================================
# AliasEntry 数据类
# ============================================================================


class TestAliasEntry:
    def test_frozen_dataclass(self):
        """AliasEntry 应为不可变"""
        entry = AliasEntry(aliases=("a", "b"), target="c", context=("x",))
        with pytest.raises(Exception):
            entry.target = "d"  # type: ignore[misc]

    def test_default_context(self):
        """context 默认为空元组"""
        entry = AliasEntry(aliases=("a",), target="b")
        assert entry.context == ()
