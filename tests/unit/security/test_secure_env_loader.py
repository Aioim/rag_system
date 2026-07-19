"""SecureEnvLoader .env 解析测试（审查 H4：export 前缀 / 行尾注释）"""
from security.secure_env_loader import SecureEnvLoader


def _parse(tmp_path, content: str) -> dict:
    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")
    loader = SecureEnvLoader(env_file)
    return loader._parse_env_lines(loader._read_env_lines())


class TestExportPrefix:
    def test_export_prefix_stripped(self, tmp_path):
        """`export KEY=value` 应解析出 KEY 而非 'export KEY'"""
        parsed = _parse(tmp_path, "export API_KEY=abc123\n")

        assert parsed == {"API_KEY": "abc123"}

    def test_export_like_variable_name_unaffected(self, tmp_path):
        """EXPORT_DIR 这类正常变量名不应被误剥离"""
        parsed = _parse(tmp_path, "EXPORT_DIR=out\n")

        assert parsed == {"EXPORT_DIR": "out"}


class TestInlineComment:
    def test_unquoted_value_inline_comment_stripped(self, tmp_path):
        """未加引号值的行尾注释应被去除"""
        parsed = _parse(tmp_path, "TIMEOUT=30 # 单位：秒\n")

        assert parsed == {"TIMEOUT": "30"}

    def test_hash_without_leading_space_kept(self, tmp_path):
        """`#` 前无空白时不视为注释（如颜色值）"""
        parsed = _parse(tmp_path, "COLOR=#ff0000\n")

        assert parsed == {"COLOR": "#ff0000"}

    def test_quoted_value_hash_kept(self, tmp_path):
        """引号包裹的值中的 ` # ` 应原样保留"""
        parsed = _parse(tmp_path, 'MOTTO="hello # world"\n')

        assert parsed == {"MOTTO": "hello # world"}
