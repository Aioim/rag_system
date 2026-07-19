"""logger 模块对 settings 热重载的跟随性测试

复现审查发现：模块级 `LogConfig = settings.log` 在 settings.reload()
后指向旧配置对象，YAML/运行时配置变更不生效。
"""

import logging

import pytest

from config import settings

# 必须在任何 settings.reload() 之前导入，确保模块级快照先于重载绑定
from logger import filters as filters_mod
from logger.core import log_exception


class _BoomLogger:
    """error() 必定抛异常的 logger，用于触发 log_exception 的兜底路径"""

    def error(self, *args, **kwargs):
        raise RuntimeError("boom")


@pytest.fixture
def reloaded_quiet_settings():
    """reload 产生新的 log 配置对象，并在新对象上开启 quiet"""
    before = settings.log
    settings.reload()
    after = settings.log
    # 前置条件：reload 会产生新的配置对象（否则本 bug 前提不成立）
    assert after is not before

    after.quiet = True
    try:
        yield after
    finally:
        settings.log.quiet = False


class TestLoggerFollowsSettingsReload:
    def test_log_exception_respects_quiet_after_reload(
        self, reloaded_quiet_settings, capsys
    ):
        """reload 后设置 quiet=True，log_exception 兜底路径不应再打印 stderr"""
        log_exception(logger_param=_BoomLogger(), exc=ValueError("x"))

        captured = capsys.readouterr()
        assert "Error in log_exception" not in captured.err

    def test_sensitive_filter_respects_quiet_after_reload(
        self, reloaded_quiet_settings, capsys, monkeypatch
    ):
        """reload 后设置 quiet=True，SensitiveDataFilter 异常兜底不应再打印 stderr"""

        def _boom(_text):
            raise RuntimeError("boom")

        monkeypatch.setattr(filters_mod, "mask_sensitive_data", _boom)

        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hello", args=None, exc_info=None,
        )
        assert filters_mod.SensitiveDataFilter().filter(record) is True

        captured = capsys.readouterr()
        assert "SensitiveDataFilter error" not in captured.err
