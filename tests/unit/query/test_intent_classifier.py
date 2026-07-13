"""IntentClassifier 测试"""
import pytest
from models.enums import Intent
from query.intent_classifier import IntentResult, IntentClassifier


class MockLLM:
    """模拟 LLM 客户端"""

    def __init__(self, response=None, should_fail=False):
        self.response = response
        self.should_fail = should_fail
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.should_fail:
            raise RuntimeError("LLM timeout")
        return self.response


@pytest.fixture
def concept_response():
    return '{"intent": "concept", "is_clear": true, "clarification_question": null}'


@pytest.fixture
def unclear_response():
    return '{"intent": "concept", "is_clear": false, "clarification_question": "您想了解哪个方面的内容？"}'


class TestIntentResult:
    def test_fields(self):
        r = IntentResult(intent=Intent.CONCEPT, is_clear=True, clarification_question=None)
        assert r.intent == Intent.CONCEPT
        assert r.is_clear is True
        assert r.clarification_question is None

    def test_with_clarification(self):
        r = IntentResult(
            intent=Intent.PROCEDURE,
            is_clear=False,
            clarification_question="请问您需要办理什么业务？",
        )
        assert r.clarification_question == "请问您需要办理什么业务？"


class TestIntentClassifierClassify:
    @pytest.mark.asyncio
    async def test_classify_concept(self, concept_response):
        llm = MockLLM(response=concept_response)
        classifier = IntentClassifier(llm)
        result = await classifier.classify("什么是RAG？")
        assert result.intent == Intent.CONCEPT
        assert result.is_clear is True
        assert result.clarification_question is None

    @pytest.mark.asyncio
    async def test_classify_unclear_query(self, unclear_response):
        llm = MockLLM(response=unclear_response)
        classifier = IntentClassifier(llm)
        result = await classifier.classify("帮帮我")
        assert result.is_clear is False
        assert result.clarification_question is not None

    @pytest.mark.asyncio
    async def test_classify_fallback_on_llm_error(self):
        llm = MockLLM(should_fail=True)
        classifier = IntentClassifier(llm)
        result = await classifier.classify("任意问题")
        # 降级：intent=concept, is_clear=True
        assert result.intent == Intent.CONCEPT
        assert result.is_clear is True
        assert result.clarification_question is None

    @pytest.mark.asyncio
    async def test_classify_fallback_on_bad_json(self):
        llm = MockLLM(response="这不是JSON")
        classifier = IntentClassifier(llm)
        result = await classifier.classify("任意问题")
        assert result.intent == Intent.CONCEPT
        assert result.is_clear is True

    @pytest.mark.asyncio
    async def test_classify_all_intents(self):
        """验证四种意图都能正确解析"""
        cases = [
            ('{"intent": "concept", "is_clear": true, "clarification_question": null}', Intent.CONCEPT),
            ('{"intent": "procedure", "is_clear": true, "clarification_question": null}', Intent.PROCEDURE),
            ('{"intent": "compare", "is_clear": true, "clarification_question": null}', Intent.COMPARE),
            ('{"intent": "lookup", "is_clear": true, "clarification_question": null}', Intent.LOOKUP),
        ]
        for response, expected_intent in cases:
            llm = MockLLM(response=response)
            classifier = IntentClassifier(llm)
            result = await classifier.classify("test")
            assert result.intent == expected_intent, f"Failed for {response}"
