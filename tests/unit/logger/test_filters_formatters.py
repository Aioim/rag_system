"""脱敏链路回归测试（审查 C1/H6）

C1: SensitiveDataFilter 改写 record.msg 会破坏 %-style 格式串
    （如 "token=%s" 被脱敏为 "token=******"，%s 占位符丢失），
    导致 getMessage() 抛 TypeError → Handler.handleError 将未脱敏的
    原始 args 打印到 stderr，且该条日志丢失。
H6: SecurityFormatter.format() 永久变异 record.msg/args，
    多 Handler 场景下游 Formatter 拿到被改写的数据。
"""

import io
import logging

import pytest

from logger.filters import SensitiveDataFilter
from logger.formatters import SecurityFormatter


@pytest.fixture
def masked_logger():
    """带 SensitiveDataFilter + SecurityFormatter 的独立 logger，输出到内存流"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        SecurityFormatter(
            SecurityFormatter.STANDARD_FORMAT, SecurityFormatter.DATE_FORMAT
        )
    )
    test_logger = logging.getLogger("test.masking.chain")
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    test_logger.addFilter(SensitiveDataFilter())
    test_logger.setLevel(logging.DEBUG)
    test_logger.propagate = False
    try:
        yield test_logger, stream
    finally:
        test_logger.removeHandler(handler)
        test_logger.filters.clear()


class TestPercentStyleMasking:
    def test_sensitive_format_string_masked_without_losing_entry(
        self, masked_logger, capsys
    ):
        """格式串含 token=%s 时：日志不丢、输出脱敏、stderr 无泄露"""
        # Arrange
        test_logger, stream = masked_logger

        # Act
        test_logger.info("login token=%s ok", "SECRETTOKEN123")

        # Assert
        out = stream.getvalue()
        err = capsys.readouterr().err
        assert "token=******" in out, "输出应包含脱敏后的 token"
        assert "SECRETTOKEN123" not in out, "明文 token 不应出现在日志输出"
        assert "login" in out and "ok" in out, "日志条目不应丢失"
        assert "Logging error" not in err, "不应触发 logging 内部错误"
        assert "SECRETTOKEN123" not in err, "明文 token 不应泄露到 stderr"

    def test_plain_message_still_masked(self, masked_logger):
        """无参数的普通消息仍应被脱敏（保持既有行为）"""
        # Arrange
        test_logger, stream = masked_logger

        # Act
        test_logger.info("user password=abc123 login")

        # Assert
        out = stream.getvalue()
        assert "password=******" in out
        assert "abc123" not in out


class TestFormatterDoesNotMutateRecord:
    def test_record_msg_and_args_restored_after_format(self):
        """format() 后 record.msg/args 应恢复原值（多 Handler 兼容）"""
        # Arrange
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="pwd=%s", args=("abc",), exc_info=None,
        )
        fmt = SecurityFormatter("%(message)s")

        # Act
        first = fmt.format(record)

        # Assert
        assert record.msg == "pwd=%s", "record.msg 不应被永久改写"
        assert record.args == ("abc",), "record.args 不应被永久改写"
        assert fmt.format(record) == first, "重复格式化结果应一致"

    def test_dict_args_restored_after_format(self):
        """dict 参数经 format() 后应恢复原值（含会被 _sanitize 改写的字符）"""
        # Arrange：值带换行符，_sanitize 会将其替换为空格
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hello %(who)s", args=({"who": "a\nb"},), exc_info=None,
        )
        fmt = SecurityFormatter("%(message)s")

        # Act
        fmt.format(record)

        # Assert
        assert record.args == {"who": "a\nb"}, "record.args 不应被永久改写"
