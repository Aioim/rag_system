"""AliasManager 测试 — 热重载与并发安全"""

import threading
import time

import yaml as yaml_mod

from config.aliases import AliasManager


class TestAliasManagerReload:
    def test_reload_picks_up_file_changes(self, tmp_path):
        """reload 后应读到文件的最新内容"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text('"工资条": "薪资明细"\n', encoding="utf-8")

        mgr = AliasManager()
        assert mgr.load(alias_file)
        assert mgr.resolve("工资条") == "薪资明细"

        alias_file.write_text('"工资条": "工资单"\n', encoding="utf-8")
        mgr.reload()

        assert mgr.resolve("工资条") == "工资单"
        assert mgr.count == 1

    def test_concurrent_resolve_during_reload_never_sees_partial_state(
        self, tmp_path, monkeypatch
    ):
        """reload 期间并发 resolve/resolve_all 不得读到半清空状态（复现热重载竞态）"""
        alias_file = tmp_path / "aliases.yaml"
        alias_file.write_text('"工资条": "薪资明细"\n', encoding="utf-8")

        mgr = AliasManager()
        assert mgr.load(alias_file)

        # 放慢 YAML 解析，放大 reload 的中间状态窗口
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
                except Exception as e:  # noqa: BLE001 — 竞态触发的任何异常都算失败
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
        # 重载完成后映射仍然完整
        assert mgr.resolve("工资条") == "薪资明细"
