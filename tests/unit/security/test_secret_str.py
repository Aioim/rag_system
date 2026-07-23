"""SecretStr 单元测试"""

import pickle

import pytest

from security import SecretStr


class TestSecretStrInit:
    def test_init_with_valid_string(self):
        s = SecretStr("my_secret_value", name="test")
        assert s.name == "test"
        assert s.get() == "my_secret_value"

    def test_init_with_empty_string(self):
        s = SecretStr("", name="empty")
        assert s.get() == ""

    def test_init_rejects_non_string(self):
        with pytest.raises(TypeError, match="Secret value must be str"):
            SecretStr(123)  # type: ignore[arg-type]

    def test_init_default_name(self):
        s = SecretStr("val")
        assert s.name == "secret"


class TestSecretStrGet:
    def test_get_returns_plain_string(self):
        s = SecretStr("hello", name="test")
        assert s.get() == "hello"
        assert isinstance(s.get(), str)

    def test_get_sets_accessed_flag(self):
        s = SecretStr("hello", name="test")
        assert not s.is_accessed()
        s.get()
        assert s.is_accessed()


class TestSecretStrMask:
    def test_mask_default_params(self):
        s = SecretStr("my_secret_password")
        masked = s.mask()
        assert masked.startswith("my_")
        assert masked.endswith("word")
        assert "my_secret_password" not in masked

    def test_mask_custom_params(self):
        s = SecretStr("abcdefghij")
        masked = s.mask(visible_start=2, visible_end=2)
        assert masked == "ab******ij"

    def test_mask_short_value(self):
        s = SecretStr("abc")
        masked = s.mask(visible_start=3, visible_end=4)
        assert len(masked) >= 6
        assert all(c == "*" for c in masked)


class TestSecretStrRepr:
    def test_repr_does_not_leak_value(self):
        s = SecretStr("my_secret", name="db")
        r = repr(s)
        assert "my_secret" not in r
        assert "db" in r
        assert "SecretStr" in r

    def test_str_does_not_leak_value(self):
        s = SecretStr("my_secret")
        assert "my_secret" not in str(s)

    def test_format_does_not_leak_value(self):
        s = SecretStr("my_secret")
        formatted = f"{s}"
        assert "my_secret" not in formatted


class TestSecretStrEquality:
    def test_eq_same_value(self):
        s1 = SecretStr("secret")
        s2 = SecretStr("secret")
        assert s1 == s2

    def test_eq_different_value(self):
        s1 = SecretStr("secret1")
        s2 = SecretStr("secret2")
        assert s1 != s2

    def test_eq_with_plain_string(self):
        s = SecretStr("secret")
        assert s == "secret"
        assert s != "other"

    def test_eq_with_non_string(self):
        s = SecretStr("secret")
        assert s != 123
        assert s != None  # noqa: E711

    def test_eq_constant_time(self):
        """验证使用 compare_digest（间接验证不会因长度差异快速返回）"""
        s = SecretStr("a" * 1000 + "x")
        # 不同长度也能正确比较
        assert s != "short"


class TestSecretStrLen:
    def test_len(self):
        assert len(SecretStr("hello")) == 5

    def test_len_empty(self):
        assert len(SecretStr("")) == 0


class TestSecretStrBool:
    def test_bool_non_empty(self):
        assert bool(SecretStr("val"))

    def test_bool_empty(self):
        assert not bool(SecretStr(""))


class TestSecretStrHash:
    def test_hash_raises_type_error(self):
        s = SecretStr("secret")
        with pytest.raises(TypeError, match="unhashable"):
            hash(s)


class TestSecretStrSerialization:
    def test_pickle_dump_raises(self):
        s = SecretStr("secret", name="test")
        # pickle 会调用 __reduce_ex__，抛出 RuntimeError
        with pytest.raises(RuntimeError, match="Cannot serialize"):
            pickle.dumps(s)

    def test_getstate_raises(self):
        s = SecretStr("secret", name="test")
        with pytest.raises(RuntimeError, match="Cannot pickle"):
            s.__getstate__()

    def test_reduce_raises(self):
        s = SecretStr("secret", name="test")
        with pytest.raises(RuntimeError, match="Cannot serialize"):
            s.__reduce__()

    def test_reduce_ex_raises(self):
        s = SecretStr("secret", name="test")
        with pytest.raises(RuntimeError, match="Cannot serialize"):
            s.__reduce_ex__(4)


class TestSecretStrOperations:
    def test_add_raises(self):
        s = SecretStr("a")
        with pytest.raises(TypeError, match="Cannot concatenate"):
            s + "b"  # type: ignore[operator]

    def test_radd_raises(self):
        s = SecretStr("a")
        with pytest.raises(TypeError, match="Cannot concatenate"):
            "b" + s  # type: ignore[operator]

    def test_mul_raises(self):
        s = SecretStr("a")
        with pytest.raises(TypeError, match="Cannot multiply"):
            s * 3  # type: ignore[operator]

    def test_rmul_raises(self):
        s = SecretStr("a")
        with pytest.raises(TypeError, match="Cannot multiply"):
            3 * s  # type: ignore[operator]


class TestSecretStrAccessFlag:
    def test_is_accessed_initially_false(self):
        s = SecretStr("secret")
        assert not s.is_accessed()

    def test_reset_access_flag(self):
        s = SecretStr("secret")
        s.get()
        assert s.is_accessed()
        s.reset_access_flag()
        assert not s.is_accessed()


class TestSecretStrTemporary:
    def test_temporary_context_manager(self):
        s = SecretStr("secret_value", name="test")
        with s.temporary() as plain:
            assert plain == "secret_value"
            assert isinstance(plain, str)

    def test_temporary_sets_accessed(self):
        s = SecretStr("secret")
        with s.temporary():
            pass
        assert s.is_accessed()
