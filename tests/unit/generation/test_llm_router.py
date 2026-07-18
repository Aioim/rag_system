"""LLMRouter 测试"""
import pytest

from config import settings
from generation.llm_router import LLMRouter, RouteResult
from models.enums import Intent


class TestRouting:
    @pytest.fixture
    def router(self):
        return LLMRouter()

    def test_lookup_routes_to_lightweight(self, router):
        result = router.route(Intent.LOOKUP)
        assert result.model_tier == "lightweight"
        assert result.model_name == settings.llm.lightweight
        assert result.temperature == 0.0

    def test_procedure_routes_to_lightweight(self, router):
        result = router.route(Intent.PROCEDURE)
        assert result.model_tier == "lightweight"
        assert result.temperature == 0.0

    def test_concept_routes_to_default(self, router):
        result = router.route(Intent.CONCEPT)
        assert result.model_tier == "default"
        assert result.model_name == settings.llm.default
        assert result.temperature == 0.3

    def test_compare_routes_to_default(self, router):
        result = router.route(Intent.COMPARE)
        assert result.model_tier == "default"
        assert result.temperature == 0.2

    def test_none_intent_falls_back_to_concept(self, router):
        result = router.route(None)
        assert result.model_tier == "default"
        assert result.temperature == 0.3


class TestTemplates:
    @pytest.fixture
    def router(self):
        return LLMRouter()

    def test_loads_template_with_placeholders(self, router):
        result = router.route(Intent.CONCEPT)
        assert result.system_prompt.strip() != ""
        assert "{context}" in result.user_template
        assert "{query}" in result.user_template

    def test_each_intent_has_own_template(self, router):
        templates = {
            intent: router.route(intent).system_prompt for intent in Intent
        }
        assert len(set(templates.values())) == len(Intent)

    def test_template_cached_across_calls(self, router):
        first = router.route(Intent.LOOKUP)
        second = router.route(Intent.LOOKUP)
        assert first.system_prompt is second.system_prompt

    def test_missing_template_raises_friendly_error(self, router, tmp_path):
        router._prompts_dir = tmp_path
        with pytest.raises(FileNotFoundError, match="Prompt 模板"):
            router.route(Intent.LOOKUP)


class TestRouteResult:
    def test_is_dataclass_with_expected_fields(self):
        result = RouteResult(
            model_tier="default",
            model_name="claude-sonnet-5",
            temperature=0.3,
            system_prompt="s",
            user_template="u",
        )
        assert result.model_name == "claude-sonnet-5"
