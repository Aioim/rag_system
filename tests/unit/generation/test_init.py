"""generation __init__ 工厂 + 单例测试"""
from generation import (
    GenerationLayer,
    CitationBuilder,
    PromptAssembler,
    LLMRouter,
    FactChecker,
    get_generation_layer,
    reset_generation_layer,
)


class TestSingleton:
    def test_get_returns_same_instance(self):
        from .conftest import MockLLM
        llm = MockLLM()
        reset_generation_layer()
        try:
            a = get_generation_layer(llm)
            b = get_generation_layer(llm)
            assert a is b
        finally:
            reset_generation_layer()

    def test_reset_creates_new_instance(self):
        from .conftest import MockLLM
        llm = MockLLM()
        reset_generation_layer()
        try:
            a = get_generation_layer(llm)
            reset_generation_layer()
            b = get_generation_layer(llm)
            assert a is not b
        finally:
            reset_generation_layer()

    def test_different_llm_returns_cached_instance(self):
        from .conftest import MockLLM
        reset_generation_layer()
        try:
            a = get_generation_layer(MockLLM())
            # 再次调用，但传入不同 llm → 返回缓存的实例
            b = get_generation_layer(MockLLM())
            assert a is b
        finally:
            reset_generation_layer()


class TestExports:
    def test_all_contains_main_types(self):
        import generation
        names = set(generation.__all__)
        assert "GenerationLayer" in names
        assert "get_generation_layer" in names
        assert "reset_generation_layer" in names
