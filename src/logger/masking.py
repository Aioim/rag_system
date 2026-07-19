"""
敏感数据脱敏引擎
提供高性能的敏感信息掩码功能，支持缓存和流式处理
"""
import re
from functools import lru_cache


class MaskingEngine:
    """高性能脱敏引擎"""

    _SHORT_THRESHOLD = 100

    # 脱敏模式库
    _PATTERNS: dict = {  # noqa: RUF012
        'auth': [
            # JSON 格式字段
            (re.compile(r'(?i)("password"\s*:\s*")[^"]+(")'), r'\1******\2'),
            (re.compile(r'(?i)("token"\s*:\s*")[^"]+(")'), r'\1******\2'),
            (re.compile(r'(?i)("api_key"\s*:\s*")[^"]+(")'), r'\1******\2'),
            (re.compile(r'(?i)("apikey"\s*:\s*")[^"]+(")'), r'\1******\2'),
            (re.compile(r'(?i)("authorization"\s*:\s*")[^"]+(")'), r'\1******\2'),
            (re.compile(r'(?i)("secret"\s*:\s*")[^"]+(")'), r'\1******\2'),
            # 查询参数格式
            (re.compile(r'(?i)(password=)[^&\s]+'), r'\1******'),
            (re.compile(r'(?i)(pwd=)[^&\s]+'), r'\1******'),
            (re.compile(r'(?i)(token=)[^&\s]+'), r'\1******'),
            (re.compile(r'(?i)(api_key=)[^&\s]+'), r'\1******'),
            (re.compile(r'(?i)(apikey=)[^&\s]+'), r'\1******'),
            (re.compile(r'(?i)(authorization=)[^&\s]+'), r'\1******'),
            (re.compile(r'(?i)(secret=)[^&\s]+'), r'\1******'),
        ],
        'pii': [
            # 电子邮箱
            (re.compile(r'(?i)([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'), r'***@\2'),
            # 手机号码（中国大陆）
            (re.compile(r'(?i)(1[3-9]\d)(\d{4})(\d{4})'), r'\1****\3'),
            # 身份证号码（18位）
            (re.compile(r'(?i)([1-9]\d{5})(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(\d{3})([0-9Xx])'),
             r'\1**********\6'),
            # 银行卡号（16-19位）
            (re.compile(r'(?i)(\d{4})(\d{8,11})(\d{4})'), r'\1******\3'),
        ],
        'credit_card': [
            # JSON 格式
            (re.compile(r'(?i)("credit_card"\s*:\s*")[^"]+(")'), r'\1******\2'),
            # 查询参数
            (re.compile(r'(?i)(credit_card=)[^&\s]+'), r'\1******'),
            # 直接卡号模式
            (re.compile(r'(?i)(\d{4})[ -]?(\d{4})[ -]?(\d{4})[ -]?(\d{4})'), r'\1**** ****\4'),
        ],
    }

    @staticmethod
    @lru_cache(maxsize=256)
    def _cached_mask(text: str) -> str:
        """
        缓存的脱敏方法（仅以原始文本为键）
        注意：lru_cache 要求参数可哈希，字符串天然满足
        """
        result = text
        for category in MaskingEngine._PATTERNS.values():
            for pattern, repl in category:
                result, n = pattern.subn(repl, result)
        return result

    @staticmethod
    def mask(text: str) -> str:
        """
        对输入文本执行脱敏操作

        Args:
            text: 待脱敏的字符串

        Returns:
            脱敏后的字符串；若输入非字符串则原样返回
        """
        if not isinstance(text, str) or not text:
            return text

        length = len(text)
        if length < MaskingEngine._SHORT_THRESHOLD:
            return MaskingEngine._direct_mask(text)
        return MaskingEngine._cached_mask(text)

    @staticmethod
    def _direct_mask(text: str) -> str:
        """直接应用所有脱敏规则（适用于短文本）"""
        result = text
        for category in MaskingEngine._PATTERNS.values():
            for pattern, repl in category:
                result, n = pattern.subn(repl, result)
        return result

def mask_sensitive_data(message):
    """
    便捷脱敏函数

    Args:
        message: 任意类型，仅当为字符串时进行脱敏

    Returns:
        脱敏后的结果（非字符串原样返回）
    """
    if isinstance(message, str):
        return MaskingEngine.mask(message)
    return message

