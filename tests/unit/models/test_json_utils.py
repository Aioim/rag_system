"""extract_json_container 单元测试"""
import pytest

from models.json_utils import extract_json_container


class TestExtractJsonContainer:
    """extract_json_container 提取 JSON 容器测试"""

    # ---- 快速路径：直接解析合法 JSON ----

    def test_direct_json_array(self):
        """整个字符串就是合法 JSON 数组时直接返回"""
        raw = '[{"claim": "test", "status": "supported"}]'
        result = extract_json_container(raw, "[", "]")
        assert result == raw

    def test_direct_json_object(self):
        """整个字符串就是合法 JSON 对象时直接返回"""
        raw = '{"intent": "concept", "is_clear": true}'
        result = extract_json_container(raw, "{", "}")
        assert result == raw

    def test_direct_json_array_with_whitespace(self):
        """前后有空白字符仍走快速路径"""
        raw = '  [{"a": 1}]  '
        result = extract_json_container(raw, "[", "]")
        assert result is not None
        # 快速路径先尝试 strip 后再解析
        assert result.strip() == raw.strip()

    # ---- 慢速路径：带前后缀文本 ----

    def test_json_with_prefix_text(self):
        """JSON 前有 Markdown 说明文本"""
        raw = 'Here is the result:\n[{"a": 1}, {"b": 2}]'
        result = extract_json_container(raw, "[", "]")
        assert result == '[{"a": 1}, {"b": 2}]'

    def test_json_with_suffix_text(self):
        """JSON 后有附加文本"""
        raw = '[{"a": 1}]\n\nHope this helps!'
        result = extract_json_container(raw, "[", "]")
        assert result == '[{"a": 1}]'

    def test_json_with_markdown_code_block(self):
        """JSON 包裹在 Markdown 代码块中"""
        raw = '```json\n[{"claim": "test"}]\n```'
        result = extract_json_container(raw, "[", "]")
        assert result == '[{"claim": "test"}]'

    def test_json_object_with_prefix(self):
        """提取 JSON 对象（花括号）"""
        raw = 'Response: {"intent": "concept", "is_clear": true}'
        result = extract_json_container(raw, "{", "}")
        assert result == '{"intent": "concept", "is_clear": true}'

    # ---- 嵌套括号 ----

    def test_nested_brackets(self):
        """正确处理嵌套方括号"""
        raw = 'Result: [[1, 2], [3, [4, 5]]]'
        result = extract_json_container(raw, "[", "]")
        assert result == '[[1, 2], [3, [4, 5]]]'

    def test_nested_braces(self):
        """正确处理嵌套花括号"""
        raw = '{"outer": {"inner": {"deep": "value"}}} extra'
        result = extract_json_container(raw, "{", "}")
        assert result == '{"outer": {"inner": {"deep": "value"}}}'

    # ---- 字符串内转义和括号 ----

    def test_escaped_quotes_in_string(self):
        """字符串内的转义引号不中断解析"""
        raw = r'[{"text": "he said \"hello\""}] trailing'
        result = extract_json_container(raw, "[", "]")
        assert result == r'[{"text": "he said \"hello\""}]'

    def test_brackets_inside_string_ignored(self):
        """字符串内的方括号不影响括号计数"""
        raw = '[{"text": "a [bracket] inside"}, {"text": "normal"}]'
        result = extract_json_container(raw, "[", "]")
        assert result == '[{"text": "a [bracket] inside"}, {"text": "normal"}]'

    def test_braces_inside_string_ignored(self):
        """字符串内的花括号不影响括号计数"""
        raw = '{"key": "value {with braces}"} extra'
        result = extract_json_container(raw, "{", "}")
        assert result == '{"key": "value {with braces}"}'

    def test_backslash_escape_in_string(self):
        """字符串内的反斜杠转义正确处理"""
        raw = r'[{"path": "C:\\Users\\test"}]'
        result = extract_json_container(raw, "[", "]")
        assert result == r'[{"path": "C:\\Users\\test"}]'

    # ---- 失败场景 ----

    def test_no_open_char_returns_none(self):
        """找不到起始括号返回 None"""
        result = extract_json_container("no brackets here at all", "[", "]")
        assert result is None

    def test_unmatched_brackets_returns_none(self):
        """括号不匹配（多余的起始括号）返回 None"""
        raw = "[[1, 2, 3"
        result = extract_json_container(raw, "[", "]")
        assert result is None

    def test_empty_input(self):
        """空字符串返回 None"""
        result = extract_json_container("", "[", "]")
        assert result is None

    def test_only_whitespace(self):
        """仅空白字符 — raw.find 找到的是 -1，返回 None"""
        result = extract_json_container("   \n\t  ", "[", "]")
        assert result is None

    # ---- 边界场景 ----

    def test_single_element_array(self):
        """单元素数组"""
        result = extract_json_container("[42]", "[", "]")
        assert result == "[42]"

    def test_empty_array(self):
        """空数组"""
        raw = "prefix [] suffix"
        result = extract_json_container(raw, "[", "]")
        assert result == "[]"

    def test_empty_object(self):
        """空对象"""
        raw = "before {} after"
        result = extract_json_container(raw, "{", "}")
        assert result == "{}"

    def test_mixed_brackets_in_same_input(self):
        """相同输入中同时存在方括号和花括号，按指定 open_char 提取"""
        raw = 'prefix {"obj": [1, 2, 3]} suffix'
        result = extract_json_container(raw, "{", "}")
        assert result == '{"obj": [1, 2, 3]}'
