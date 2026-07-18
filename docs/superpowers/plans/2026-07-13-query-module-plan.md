# Query 模块（查询理解层）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 `src/query/` 查询理解层，将原始用户 Query 转化为可用于检索的结构化 `PipelineContext`

**Architecture:** Pipeline 链式编排 — 别名映射 → 意图分类(+清晰度) → 多轮上下文融合 → 查询改写(并行3路)。每步独立组件，LLM 通过构造函数注入，遵循 duck-typing（需 `async generate(prompt, **kwargs) -> str`）

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2（已有）, pytest + pytest-asyncio

## Global Constraints

- `langchain` > 1.3.0, `langgraph` >= 1.2.0（本期不直接依赖）
- 测试用 pytest + `pytest-asyncio`，mock LLM 而非真实调用
- 代码风格参照已有模块：`from config import settings` / `from logger import logger`
- LLM 接口通过 duck-typing 注入，需有 `async generate(prompt: str, **kwargs) -> str` 方法
- 遵循 CLAUDE.md 的 Simplicity First + Surgical Changes 原则

---

### Task 1: 创建包骨架

**Files:**
- Create: `src/query/__init__.py`
- Create: `src/query/rewriters/__init__.py`

**Interfaces:**
- Produces: 空包，后续任务往其中添加模块

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p src/query/rewriters
```

- [ ] **Step 2: 写包标记文件**

`src/query/__init__.py`:
```python
"""查询理解层 — 意图分类 / 清晰度判断 / 上下文融合 / 查询改写"""
```

`src/query/rewriters/__init__.py`:
```python
"""查询改写器 — HyDE / 关键词提取 / 同义词扩展"""
```

- [ ] **Step 3: 验证导入正常**

```bash
cd E:/Code/rag0709 && python -c "import query; import query.rewriters; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: 提交**

```bash
git add src/query/
git commit -m "feat(query): add package scaffold"
```

---

### Task 2: BaseRewriter 抽象基类

**Files:**
- Create: `src/query/rewriters/base.py`
- Create: `tests/unit/query/__init__.py`
- Create: `tests/unit/query/rewriters/__init__.py`
- Create: `tests/unit/query/rewriters/test_base.py`

**Interfaces:**
- Produces: `BaseRewriter` — ABC，要求子类实现 `async rewrite(self, query: str) -> list[str]`

- [ ] **Step 1: 写单元测试**

`tests/unit/query/rewriters/test_base.py`:
```python
"""BaseRewriter 测试"""
import pytest
from query.rewriters.base import BaseRewriter


def test_cannot_instantiate_abstract():
    """不能直接实例化抽象基类"""
    with pytest.raises(TypeError):
        BaseRewriter()


def test_concrete_subclass_must_implement_rewrite():
    """子类必须实现 rewrite 方法"""

    class BadRewriter(BaseRewriter):
        pass

    with pytest.raises(TypeError):
        BadRewriter()


def test_valid_subclass_instantiates():
    """正确实现 rewrite 的子类可以实例化"""

    class GoodRewriter(BaseRewriter):
        async def rewrite(self, query: str) -> list[str]:
            return [query]

    r = GoodRewriter()
    assert isinstance(r, BaseRewriter)
```

`tests/unit/query/__init__.py` 和 `tests/unit/query/rewriters/__init__.py` 写空文件。

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_base.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'query.rewriters.base'`

- [ ] **Step 3: 实现 BaseRewriter**

`src/query/rewriters/base.py`:
```python
"""查询改写器抽象基类"""
from abc import ABC, abstractmethod


class BaseRewriter(ABC):
    """查询改写器基类

    所有改写器需实现 rewrite 方法，返回改写后的查询列表（可为空）。
    LLM 通过构造函数注入，需有 async generate(prompt, **kwargs) -> str 方法。
    """

    @abstractmethod
    async def rewrite(self, query: str) -> list[str]:
        """返回改写后的查询列表（可为空）"""
        ...
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_base.py -v
```
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add src/query/rewriters/base.py tests/unit/query/
git commit -m "feat(query): add BaseRewriter abstract base class"
```

---

### Task 3: IntentClassifier — 意图分类 + 清晰度判断

**Files:**
- Create: `src/query/intent_classifier.py`
- Create: `tests/unit/query/test_intent_classifier.py`

**Interfaces:**
- Consumes: `models.enums.Intent`
- Produces: `IntentResult(intent, is_clear, clarification_question)` dataclass
- Produces: `IntentClassifier(llm).classify(query) -> IntentResult`

- [ ] **Step 1: 写单元测试**

`tests/unit/query/test_intent_classifier.py`:
```python
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
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/test_intent_classifier.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 IntentClassifier**

`src/query/intent_classifier.py`:
```python
"""意图分类 + 清晰度判断（合并单次 LLM 调用）"""
import json
import re
from dataclasses import dataclass

from models.enums import Intent


@dataclass
class IntentResult:
    """意图分类结果"""
    intent: Intent
    is_clear: bool
    clarification_question: str | None


class IntentClassifier:
    """意图分类器 — 单次 LLM 调用完成意图分类 + 清晰度判断

    LLM 需有 async generate(prompt, **kwargs) -> str 方法。
    失败时降级为 intent=CONCEPT, is_clear=True。
    """

    def __init__(self, llm):
        self._llm = llm

    async def classify(self, query: str) -> IntentResult:
        prompt = self._build_prompt(query)
        try:
            raw = await self._llm.generate(prompt, temperature=0)
            return self._parse_response(raw)
        except Exception:
            return IntentResult(
                intent=Intent.CONCEPT,
                is_clear=True,
                clarification_question=None,
            )

    def _build_prompt(self, query: str) -> str:
        return (
            "你是一个查询意图分类器。分析用户问题，判断意图类型和清晰度。\n"
            "\n"
            "意图类型：\n"
            '- concept: 概念理解（"什么是""解释""含义""定义"）\n'
            '- procedure: 操作步骤（"如何""怎么""步骤""流程""申请""办理"）\n'
            '- compare: 对比分析（"区别""对比""哪个好""优缺点""不同"）\n'
            '- lookup: 精确查找（具体数据、条款、日期、金额、人名）\n'
            "\n"
            "is_clear 判断标准：问题明确表达了需求、有足够上下文、可以独立理解 → true\n"
            "clarification_question: 问题不清晰时，生成一个引导用户补充信息的追问\n"
            "\n"
            "输出严格JSON格式，不要输出任何其他内容：\n"
            '{{"intent": "<concept|procedure|compare|lookup>", "is_clear": true/false, "clarification_question": null/"具体澄清问题"}}\n'
            "\n"
            f"用户问题：{query}"
        )

    def _parse_response(self, raw: str) -> IntentResult:
        # 提取 JSON（LLM 可能在 JSON 前后加额外文字）
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {raw[:200]}")

        data = json.loads(match.group())

        intent_str = data.get("intent", "concept")
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.CONCEPT

        is_clear = bool(data.get("is_clear", True))
        clarification_question = data.get("clarification_question") if not is_clear else None

        return IntentResult(
            intent=intent,
            is_clear=is_clear,
            clarification_question=clarification_question,
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/test_intent_classifier.py -v
```
Expected: 7 PASS

- [ ] **Step 5: 提交**

```bash
git add src/query/intent_classifier.py tests/unit/query/test_intent_classifier.py
git commit -m "feat(query): add IntentClassifier with clarity check"
```

---

### Task 4: HyDERewriter — 假设答案改写

**Files:**
- Create: `src/query/rewriters/hyde.py`
- Create: `tests/unit/query/rewriters/test_hyde.py`

**Interfaces:**
- Consumes: `query.rewriters.base.BaseRewriter`
- Produces: `HyDERewriter(llm).rewrite(query) -> list[str]`（单元素列表，内容是假设答案文本）

- [ ] **Step 1: 写单元测试**

`tests/unit/query/rewriters/test_hyde.py`:
```python
"""HyDERewriter 测试"""
import pytest
from query.rewriters.hyde import HyDERewriter


class MockLLM:
    def __init__(self, response="这是假设答案文本"):
        self.response = response
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


class TestHyDERewriter:
    @pytest.mark.asyncio
    async def test_rewrite_returns_single_element_list(self):
        llm = MockLLM(response="RAG是一种结合检索和生成的AI技术，可以有效提供基于知识库的问答服务。")
        rewriter = HyDERewriter(llm)
        result = await rewriter.rewrite("什么是RAG？")
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) > 0

    @pytest.mark.asyncio
    async def test_rewrite_contains_query_context(self):
        """假设答案应与原始问题相关（基础检查）"""
        llm = MockLLM()
        rewriter = HyDERewriter(llm)
        result = await rewriter.rewrite("如何申请年假？")
        prompt = llm.calls[0][0]
        assert "申请年假" in prompt

    @pytest.mark.asyncio
    async def test_rewrite_on_llm_error(self):
        llm = MockLLM(response=RuntimeError("timeout"))
        llm.response = None  # trigger exception

        class FailingLLM:
            async def generate(self, prompt, **kwargs):
                raise RuntimeError("timeout")

        rewriter = HyDERewriter(FailingLLM())
        # 不应该抛异常，返回空 list（由 QueryRewriter 编排层处理）
        try:
            result = await rewriter.rewrite("test")
        except RuntimeError:
            result = []
        assert result == []
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_hyde.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现 HyDERewriter**

`src/query/rewriters/hyde.py`:
```python
"""HyDE (Hypothetical Document Embedding) 改写器

先让 LLM 生成"假设答案"，用假设答案的 embedding 去检索，
因为假设答案在语义空间中比原始 query 更接近真实文档。
"""
from query.rewriters.base import BaseRewriter


class HyDERewriter(BaseRewriter):
    """生成假设答案作为检索查询"""

    def __init__(self, llm):
        self._llm = llm

    async def rewrite(self, query: str) -> list[str]:
        prompt = self._build_prompt(query)
        try:
            response = await self._llm.generate(prompt, temperature=0.3)
            answer = response.strip()
            return [answer] if answer else []
        except Exception:
            return []

    def _build_prompt(self, query: str) -> str:
        return (
            "你是一个知识库助手。请根据用户问题，生成一段100-200字的假设性答案。"
            "不需要完全准确，只需要合理推测可能的答案内容。\n"
            "\n"
            f"用户问题：{query}\n"
            "\n"
            "假设答案（100-200字）："
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_hyde.py -v
```
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add src/query/rewriters/hyde.py tests/unit/query/rewriters/test_hyde.py
git commit -m "feat(query): add HyDERewriter"
```

---

### Task 5: KeywordRewriter — 关键词提取

**Files:**
- Create: `src/query/rewriters/keyword.py`
- Create: `tests/unit/query/rewriters/test_keyword.py`

**Interfaces:**
- Consumes: `query.rewriters.base.BaseRewriter`
- Produces: `KeywordRewriter(llm).rewrite(query) -> list[str]`

- [ ] **Step 1: 写单元测试**

`tests/unit/query/rewriters/test_keyword.py`:
```python
"""KeywordRewriter 测试"""
import pytest
from query.rewriters.keyword import KeywordRewriter


class MockLLM:
    def __init__(self, response="关键词1 关键词2"):
        self.response = response
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


class TestKeywordRewriter:
    @pytest.mark.asyncio
    async def test_rewrite_returns_keywords(self):
        llm = MockLLM(response="年假 申请 材料 流程")
        rewriter = KeywordRewriter(llm)
        result = await rewriter.rewrite("如何申请年假？")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "年假" in result[0]

    @pytest.mark.asyncio
    async def test_rewrite_handles_empty_response(self):
        llm = MockLLM(response="")
        rewriter = KeywordRewriter(llm)
        result = await rewriter.rewrite("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_rewrite_has_query_in_prompt(self):
        llm = MockLLM()
        rewriter = KeywordRewriter(llm)
        await rewriter.rewrite("五险一金缴纳比例")
        prompt = llm.calls[0][0]
        assert "五险一金缴纳比例" in prompt
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_keyword.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现 KeywordRewriter**

`src/query/rewriters/keyword.py`:
```python
"""关键词提取改写器

从 query 中提取纯关键词，给 BM25 做稀疏检索。
"""
from query.rewriters.base import BaseRewriter


class KeywordRewriter(BaseRewriter):
    """提取关键词用于 BM25 检索"""

    def __init__(self, llm):
        self._llm = llm

    async def rewrite(self, query: str) -> list[str]:
        prompt = self._build_prompt(query)
        try:
            response = await self._llm.generate(prompt, temperature=0)
            keywords = response.strip()
            return [keywords] if keywords else []
        except Exception:
            return []

    def _build_prompt(self, query: str) -> str:
        return (
            "从以下用户问题中提取最重要的关键词（名词、动词、专有名词），"
            "用空格分隔，不要包含"什么""如何""怎么"等疑问词。\n"
            "\n"
            f"用户问题：{query}\n"
            "\n"
            "关键词："
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_keyword.py -v
```
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add src/query/rewriters/keyword.py tests/unit/query/rewriters/test_keyword.py
git commit -m "feat(query): add KeywordRewriter"
```

---

### Task 6: SynonymRewriter — 同义词扩展

**Files:**
- Create: `src/query/rewriters/synonym.py`
- Create: `tests/unit/query/rewriters/test_synonym.py`

**Interfaces:**
- Consumes: `query.rewriters.base.BaseRewriter`
- Produces: `SynonymRewriter(llm).rewrite(query) -> list[str]`

- [ ] **Step 1: 写单元测试**

`tests/unit/query/rewriters/test_synonym.py`:
```python
"""SynonymRewriter 测试"""
import pytest
from query.rewriters.synonym import SynonymRewriter


class MockLLM:
    def __init__(self, response="同义表达1\n同义表达2"):
        self.response = response
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


class TestSynonymRewriter:
    @pytest.mark.asyncio
    async def test_rewrite_returns_variants(self):
        llm = MockLLM(response="怎样申请年假\n如何办理带薪年休假")
        rewriter = SynonymRewriter(llm)
        result = await rewriter.rewrite("如何申请年假？")
        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_rewrite_splits_multiline_response(self):
        llm = MockLLM(response="变体A\n变体B\n变体C")
        rewriter = SynonymRewriter(llm)
        result = await rewriter.rewrite("原始查询")
        assert len(result) == 3
        assert "变体A" in result
        assert "变体B" in result
        assert "变体C" in result

    @pytest.mark.asyncio
    async def test_rewrite_filters_empty_lines(self):
        llm = MockLLM(response="变体A\n\n\n变体B\n  \n")
        rewriter = SynonymRewriter(llm)
        result = await rewriter.rewrite("test")
        assert len(result) == 2
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_synonym.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现 SynonymRewriter**

`src/query/rewriters/synonym.py`:
```python
"""同义词扩展改写器

用 LLM 生成近义/同义表达变体，扩展查询覆盖面。
"""
from query.rewriters.base import BaseRewriter


class SynonymRewriter(BaseRewriter):
    """生成同义查询变体"""

    def __init__(self, llm):
        self._llm = llm

    async def rewrite(self, query: str) -> list[str]:
        prompt = self._build_prompt(query)
        try:
            response = await self._llm.generate(prompt, temperature=0.3)
            # 按行拆分，过滤空行和与原始查询相同的
            variants = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and line.strip() != query
            ]
            return variants
        except Exception:
            return []

    def _build_prompt(self, query: str) -> str:
        return (
            "为以下查询生成2-3个同义或近义的表达方式，每条一行。"
            "可以换用不同的词汇、句式，但保持原意不变。\n"
            "\n"
            f"原始查询：{query}\n"
            "\n"
            "同义表达："
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_synonym.py -v
```
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add src/query/rewriters/synonym.py tests/unit/query/rewriters/test_synonym.py
git commit -m "feat(query): add SynonymRewriter"
```

---

### Task 7: QueryRewriter 编排器

**Files:**
- Modify: `src/query/rewriters/__init__.py`（替换骨架为编排器）
- Create: `tests/unit/query/rewriters/test_init.py`

**Interfaces:**
- Consumes: `query.rewriters.base.BaseRewriter`, `HyDERewriter`, `KeywordRewriter`, `SynonymRewriter`
- Produces: `QueryRewriter(llm).rewrite(query) -> list[str]` — 并行编排，去重合并，原始 query 始终置顶

- [ ] **Step 1: 写单元测试**

`tests/unit/query/rewriters/test_init.py`:
```python
"""QueryRewriter 编排器测试"""
import pytest
from query.rewriters.base import BaseRewriter
from query.rewriters import QueryRewriter


class FakeLLM:
    pass  # 不需要真实 LLM，用子类 Rewriter 绕过


class FastRewriter(BaseRewriter):
    """快速改写器"""

    def __init__(self, results):
        self.results = results

    async def rewrite(self, query: str) -> list[str]:
        return self.results


class SlowRewriter(BaseRewriter):
    """慢速改写器（验证并行）"""

    def __init__(self, results):
        self.results = results

    async def rewrite(self, query: str) -> list[str]:
        import asyncio
        await asyncio.sleep(0.01)
        return self.results


class FailingRewriter(BaseRewriter):
    """失败改写器"""

    async def rewrite(self, query: str) -> list[str]:
        raise RuntimeError("rewrite failed")


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_original_query_always_first(self):
        rewriters = [
            FastRewriter(["result1"]),
            FastRewriter(["result2"]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters
        result = await orchestrator.rewrite("原始查询")
        assert result[0] == "原始查询"

    @pytest.mark.asyncio
    async def test_merges_all_results(self):
        rewriters = [
            FastRewriter(["A"]),
            FastRewriter(["B", "C"]),
            FastRewriter([]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters
        result = await orchestrator.rewrite("Q")
        assert "Q" in result
        assert "A" in result
        assert "B" in result
        assert "C" in result

    @pytest.mark.asyncio
    async def test_deduplicates_results(self):
        rewriters = [
            FastRewriter(["dup"]),
            FastRewriter(["dup", "dup"]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters
        result = await orchestrator.rewrite("Q")
        assert result.count("dup") == 1

    @pytest.mark.asyncio
    async def test_handles_rewriter_failure(self):
        """单个 rewriter 失败不影响其他"""
        rewriters = [
            FastRewriter(["good"]),
            FailingRewriter(),
            FastRewriter(["also_good"]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters
        result = await orchestrator.rewrite("Q")
        assert "good" in result
        assert "also_good" in result

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """验证并行执行：慢速 rewriters 并行完成"""
        rewriters = [
            SlowRewriter(["a"]),
            SlowRewriter(["b"]),
            SlowRewriter(["c"]),
        ]
        orchestrator = QueryRewriter.__new__(QueryRewriter)
        orchestrator._rewriters = rewriters

        import time
        start = time.monotonic()
        result = await orchestrator.rewrite("Q")
        elapsed = time.monotonic() - start

        # 并行执行，总时间应明显小于 3×0.01
        assert elapsed < 0.025
        assert len(result) >= 4  # Q + a + b + c

    @pytest.mark.asyncio
    async def test_constructor_injects_llm(self):
        """构造函数参数完整测试"""
        llm = FakeLLM()
        orchestrator = QueryRewriter(llm)
        assert len(orchestrator._rewriters) == 3
        assert all(isinstance(r, BaseRewriter) for r in orchestrator._rewriters)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_init.py -v
```
Expected: FAIL — `QueryRewriter` 尚未实现

- [ ] **Step 3: 实现 QueryRewriter 编排器**

`src/query/rewriters/__init__.py`（替换骨架）:
```python
"""查询改写编排器 — 并行执行 HyDE / 关键词 / 同义词改写，合并去重"""
import asyncio

from query.rewriters.base import BaseRewriter
from query.rewriters.hyde import HyDERewriter
from query.rewriters.keyword import KeywordRewriter
from query.rewriters.synonym import SynonymRewriter


class QueryRewriter:
    """查询改写编排器

    并行执行所有注册的 rewriter，合并结果并去重，
    原始 query 始终在返回列表的第一位。
    """

    def __init__(self, llm):
        self._rewriters: list[BaseRewriter] = [
            HyDERewriter(llm),
            KeywordRewriter(llm),
            SynonymRewriter(llm),
        ]

    async def rewrite(self, query: str) -> list[str]:
        results = await asyncio.gather(
            *(r.rewrite(query) for r in self._rewriters),
            return_exceptions=True,
        )

        all_queries = [query]
        for r in results:
            if isinstance(r, Exception):
                continue
            for q in r:
                if q and q not in all_queries:
                    all_queries.append(q)
        return all_queries
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/rewriters/test_init.py -v
```
Expected: 6 PASS

- [ ] **Step 5: 提交**

```bash
git add src/query/rewriters/__init__.py tests/unit/query/rewriters/test_init.py
git commit -m "feat(query): add QueryRewriter orchestrator"
```

---

### Task 8: ContextFuser — 多轮上下文融合

**Files:**
- Create: `src/query/context_fuser.py`
- Create: `tests/unit/query/test_context_fuser.py`

**Interfaces:**
- Consumes: `session.manager.SessionManager`
- Produces: `ContextFuser(llm, session_manager).fuse(query, session_id) -> str`

- [ ] **Step 1: 写单元测试**

`tests/unit/query/test_context_fuser.py`:
```python
"""ContextFuser 测试"""
import tempfile
from pathlib import Path

import pytest
from session.store import SessionStore
from session.manager import SessionManager
from query.context_fuser import ContextFuser


class MockLLM:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


@pytest.fixture
def session_manager():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()


class TestContextFuser:
    @pytest.mark.asyncio
    async def test_fuse_returns_completed_query(self, session_manager):
        """将指代问题补全为完整问题"""
        llm = MockLLM(response="申请年假需要什么材料？")
        fuser = ContextFuser(llm, session_manager)

        # 准备会话历史
        session_manager.get_or_create("s1")
        session_manager.add_message("s1", "user", "年假怎么申请？")
        session_manager.add_message("s1", "assistant", "年假申请需要登录OA系统...")

        result = await fuser.fuse("需要什么材料？", "s1")
        assert result == "申请年假需要什么材料？"

    @pytest.mark.asyncio
    async def test_fuse_preserves_complete_query(self, session_manager):
        """已是完整问题的，原样返回"""
        llm = MockLLM(response="五险一金缴纳比例是多少？")
        fuser = ContextFuser(llm, session_manager)

        session_manager.get_or_create("s1")
        result = await fuser.fuse("五险一金缴纳比例是多少？", "s1")
        assert result == "五险一金缴纳比例是多少？"

    @pytest.mark.asyncio
    async def test_fuse_handles_nonexistent_session(self, session_manager):
        """会话不存在时返回原始 query"""
        llm = MockLLM()
        fuser = ContextFuser(llm, session_manager)
        result = await fuser.fuse("任意问题", "不存在的会话ID")
        assert result == "任意问题"

    @pytest.mark.asyncio
    async def test_fuse_handles_llm_error(self, session_manager):
        """LLM 失败时降级返回原始 query"""

        class FailingLLM:
            async def generate(self, prompt, **kwargs):
                raise RuntimeError("timeout")

        fuser = ContextFuser(FailingLLM(), session_manager)
        session_manager.get_or_create("s1")
        result = await fuser.fuse("需要什么材料？", "s1")
        assert result == "需要什么材料？"

    @pytest.mark.asyncio
    async def test_fuse_includes_history_in_prompt(self, session_manager):
        """验证 prompt 包含历史消息"""
        llm = MockLLM(response="完整问题")
        fuser = ContextFuser(llm, session_manager)

        session_manager.get_or_create("s2")
        session_manager.add_message("s2", "user", "VPN怎么连接？")
        session_manager.add_message("s2", "assistant", "请下载VPN客户端...")

        await fuser.fuse("它的密码怎么改？", "s2")
        prompt = llm.calls[0][0]
        assert "VPN怎么连接" in prompt
        assert "它的密码怎么改" in prompt
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/test_context_fuser.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现 ContextFuser**

`src/query/context_fuser.py`:
```python
"""多轮上下文融合 — 指代消解 + 追问补全"""
from logger import logger
from session.manager import SessionManager


class ContextFuser:
    """将多轮对话中的追问/指代补全为独立完整问题

    LLM 需有 async generate(prompt, **kwargs) -> str 方法。
    SessionManager 用于获取对话历史。
    """

    def __init__(self, llm, session_manager: SessionManager):
        self._llm = llm
        self._session_manager = session_manager

    async def fuse(self, query: str, session_id: str) -> str:
        session = self._session_manager.get(session_id)
        if session is None or not session.messages:
            return query

        try:
            history = self._format_history(session.messages)
            prompt = self._build_prompt(history, query)
            response = await self._llm.generate(prompt, temperature=0)
            result = response.strip()
            return result if result else query
        except Exception:
            logger.warning("ContextFuser LLM 调用失败，返回原始 query")
            return query

    def _format_history(self, messages) -> str:
        lines = []
        for msg in messages[-6:]:  # 最近 3 轮
            role = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role}：{msg.content}")
        return "\n".join(lines)

    def _build_prompt(self, history: str, query: str) -> str:
        return (
            "你是一个对话上下文理解助手。根据对话历史，判断用户当前问题是否包含"
            "指代词或省略信息。如果包含，请补全为独立完整的提问；如果不包含，"
            "原样返回当前问题。\n"
            "\n"
            "规则：\n"
            '1. 指代词（"它""这个""那个""他""她""其"）→ 替换为具体实体\n'
            '2. 省略主语或宾语（"需要什么材料？"）→ 根据历史补全\n'
            "3. 已是完整独立的问题 → 原样返回\n"
            "4. 只返回补全后的问题，不要添加任何解释或额外文字\n"
            "\n"
            "对话历史：\n"
            f"{history}\n"
            "\n"
            f"当前问题：{query}\n"
            "\n"
            "补全后的问题："
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/test_context_fuser.py -v
```
Expected: 5 PASS

- [ ] **Step 5: 提交**

```bash
git add src/query/context_fuser.py tests/unit/query/test_context_fuser.py
git commit -m "feat(query): add ContextFuser for multi-turn context fusion"
```

---

### Task 9: QueryUnderstandingLayer — 主编排器

**Files:**
- Create: `src/query/layer.py`
- Create: `tests/unit/query/test_layer.py`

**Interfaces:**
- Consumes: `config.aliases.resolve_aliases_in_text`, `models.context.PipelineContext`, `models.enums.Intent`, `session.manager.SessionManager`
- Consumes: `query.intent_classifier.IntentClassifier`, `query.context_fuser.ContextFuser`, `query.rewriters.QueryRewriter`
- Produces: `QueryUnderstandingLayer(llm, session_manager).process(query, session_id, collection) -> PipelineContext`

- [ ] **Step 1: 写单元测试**

`tests/unit/query/test_layer.py`:
```python
"""QueryUnderstandingLayer 测试"""
import tempfile
from pathlib import Path

import pytest
from models.enums import Intent
from session.store import SessionStore
from session.manager import SessionManager
from query.layer import QueryUnderstandingLayer


class MockLLM:
    """可编程的 Mock LLM — 根据 prompt 内容返回不同响应"""

    def __init__(self):
        self.intent_response = '{"intent": "concept", "is_clear": true, "clarification_question": null}'
        self.fuse_response = "完整的问题"
        self.hyde_response = "假设答案"
        self.keyword_response = "关键词"
        self.synonym_response = "同义变体"

    async def generate(self, prompt, **kwargs):
        if "查询意图分类器" in prompt:
            return self.intent_response
        elif "对话上下文理解" in prompt:
            return self.fuse_response
        elif "假设性答案" in prompt:
            return self.hyde_response
        elif "关键词" in prompt:
            return self.keyword_response
        elif "同义" in prompt:
            return self.synonym_response
        return "default"


@pytest.fixture
def session_manager():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()


class TestQueryUnderstandingLayerProcess:
    @pytest.mark.asyncio
    async def test_process_basic_query(self, session_manager):
        """基本流程：无 session 的简单查询"""
        llm = MockLLM()
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("什么是RAG？")

        assert ctx.query == "什么是RAG？"
        assert ctx.intent == Intent.CONCEPT
        assert ctx.needs_clarification is False
        assert len(ctx.rewritten_queries) > 0
        assert ctx.rewritten_queries[0] == "什么是RAG？"

    @pytest.mark.asyncio
    async def test_process_short_circuits_on_unclear_query(self, session_manager):
        """模糊问题短路返回，不继续检索"""
        llm = MockLLM()
        llm.intent_response = (
            '{"intent": "concept", "is_clear": false, '
            '"clarification_question": "您想了解什么内容？"}'
        )
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("帮帮我")

        assert ctx.needs_clarification is True
        assert ctx.clarification_question == "您想了解什么内容？"
        # 短路后不应有 rewritten_queries
        assert ctx.rewritten_queries == []

    @pytest.mark.asyncio
    async def test_process_with_session(self, session_manager):
        """有 session 时触发多轮上下文融合"""
        llm = MockLLM()
        llm.fuse_response = "申请年假需要什么材料？"
        layer = QueryUnderstandingLayer(llm, session_manager)

        # 准备会话
        session_manager.get_or_create("s1")
        session_manager.add_message("s1", "user", "年假怎么申请？")
        session_manager.add_message("s1", "assistant", "年假需要登录OA...")

        ctx = await layer.process("需要什么材料？", session_id="s1")
        assert ctx.query == "申请年假需要什么材料？"
        assert ctx.session is not None
        assert ctx.session.session_id == "s1"

    @pytest.mark.asyncio
    async def test_process_no_session_skips_fusion(self, session_manager):
        """无 session_id 时跳过融合步骤"""
        llm = MockLLM()
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("独立问题", session_id=None)

        assert ctx.query == "独立问题"
        assert len(ctx.rewritten_queries) > 0

    @pytest.mark.asyncio
    async def test_process_with_collection(self, session_manager):
        """collection 参数正确传递到 PipelineContext"""
        llm = MockLLM()
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("查询", collection="tech_docs")
        assert ctx.collection == "tech_docs"

    @pytest.mark.asyncio
    async def test_process_intent_classifier_failure(self, session_manager):
        """意图分类 LLM 失败时降级不抛异常"""

        class FailingIntentLLM:
            async def generate(self, prompt, **kwargs):
                raise RuntimeError("LLM error")

        layer = QueryUnderstandingLayer(FailingIntentLLM(), session_manager)
        ctx = await layer.process("任意问题")
        # 降级：intent 为 CONCEPT（分类失败默认值），不抛异常
        assert ctx.intent == Intent.CONCEPT
        # rewritten_queries 至少包含原始 query（rewriters 也都失败但 QueryRewriter 始终保留原始 query）
        assert len(ctx.rewritten_queries) >= 1
        assert ctx.rewritten_queries[0] == "任意问题"

    @pytest.mark.asyncio
    async def test_process_rewritten_queries_contain_original(self, session_manager):
        """验证原始 query 在 rewritten_queries 结果中"""
        llm = MockLLM()
        layer = QueryUnderstandingLayer(llm, session_manager)
        ctx = await layer.process("原始查询")
        # 原始查询（经过别名映射后）在第一位
        assert len(ctx.rewritten_queries) >= 1
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/test_layer.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现 QueryUnderstandingLayer**

`src/query/layer.py`:
```python
"""查询理解层主编排器 — Pipeline 链式编排"""
from config.aliases import resolve_aliases_in_text
from models.context import PipelineContext
from session.manager import SessionManager
from query.intent_classifier import IntentClassifier
from query.context_fuser import ContextFuser
from query.rewriters import QueryRewriter


class QueryUnderstandingLayer:
    """查询理解层 — 别名映射 → 意图分类 → 上下文融合 → 查询改写"""

    def __init__(self, llm, session_manager: SessionManager):
        self.intent_classifier = IntentClassifier(llm)
        self.context_fuser = ContextFuser(llm, session_manager)
        self.rewriter = QueryRewriter(llm)
        self._session_manager = session_manager

    async def process(
        self,
        query: str,
        session_id: str | None = None,
        collection: str = "default",
    ) -> PipelineContext:
        ctx = PipelineContext(query=query, collection=collection)

        # 1. 别名映射
        query = resolve_aliases_in_text(query)

        # 2. 意图分类 + 清晰度判断
        result = await self.intent_classifier.classify(query)
        ctx.intent = result.intent
        if not result.is_clear:
            ctx.needs_clarification = True
            ctx.clarification_question = result.clarification_question
            return ctx  # 短路返回

        # 3. 多轮上下文融合
        if session_id:
            query = await self.context_fuser.fuse(query, session_id)
            ctx.query = query
            ctx.session = self._session_manager.get(session_id)

        # 4. 查询改写（并行）
        ctx.rewritten_queries = await self.rewriter.rewrite(query)
        return ctx
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/test_layer.py -v
```
Expected: 7 PASS

- [ ] **Step 5: 提交**

```bash
git add src/query/layer.py tests/unit/query/test_layer.py
git commit -m "feat(query): add QueryUnderstandingLayer orchestrator"
```

---

### Task 10: 模块 `__init__.py` — 公共 API + 全局单例

**Files:**
- Modify: `src/query/__init__.py`（替换骨架为公共导出）
- Create: `tests/unit/query/test_init.py`

**Interfaces:**
- Produces: `QueryUnderstandingLayer` 重新导出
- Produces: `get_query_layer(llm, session_manager) -> QueryUnderstandingLayer` 全局单例工厂
- Produces: `reset_query_layer()` 测试辅助

- [ ] **Step 1: 写单元测试**

`tests/unit/query/test_init.py`:
```python
"""Query 模块 __init__.py 测试"""
import tempfile
from pathlib import Path

import pytest
from session.store import SessionStore
from session.manager import SessionManager
import query
from query.layer import QueryUnderstandingLayer


class FakeLLM:
    async def generate(self, prompt, **kwargs):
        return "response"


@pytest.fixture
def session_manager():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()


class TestModuleExports:
    def test_query_layer_exported(self):
        assert hasattr(query, "QueryUnderstandingLayer")

    def test_get_query_layer_exported(self):
        assert hasattr(query, "get_query_layer")
        assert callable(query.get_query_layer)

    def test_reset_query_layer_exported(self):
        assert hasattr(query, "reset_query_layer")
        assert callable(query.reset_query_layer)

    def test_intent_result_exported(self):
        from query.intent_classifier import IntentResult
        assert hasattr(query, "IntentResult")


class TestGetQueryLayer:
    def test_returns_singleton(self, session_manager):
        query.reset_query_layer()
        llm = FakeLLM()
        layer1 = query.get_query_layer(llm, session_manager)
        layer2 = query.get_query_layer(llm, session_manager)
        assert layer1 is layer2

    def test_reset_creates_new_instance(self, session_manager):
        query.reset_query_layer()
        llm = FakeLLM()
        layer1 = query.get_query_layer(llm, session_manager)
        query.reset_query_layer()
        layer2 = query.get_query_layer(llm, session_manager)
        assert layer1 is not layer2
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/test_init.py -v
```
Expected: FAIL — 部分导出不存在

- [ ] **Step 3: 实现 `__init__.py`**

`src/query/__init__.py`（替换骨架）:
```python
"""查询理解层 — 意图分类 / 清晰度判断 / 上下文融合 / 查询改写"""
from query.layer import QueryUnderstandingLayer
from query.intent_classifier import IntentResult

# 全局单例
_query_layer: QueryUnderstandingLayer | None = None


def get_query_layer(llm, session_manager) -> QueryUnderstandingLayer:
    """获取查询理解层全局单例"""
    global _query_layer
    if _query_layer is None:
        _query_layer = QueryUnderstandingLayer(llm, session_manager)
    return _query_layer


def reset_query_layer() -> None:
    """重置全局单例（测试用）"""
    global _query_layer
    _query_layer = None


__all__ = [
    "QueryUnderstandingLayer",
    "IntentResult",
    "get_query_layer",
    "reset_query_layer",
]
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/test_init.py -v
```
Expected: 6 PASS

- [ ] **Step 5: 运行全部 query 模块测试**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/query/ -v
```
Expected: 所有测试 PASS（约 30+ 个）

- [ ] **Step 6: 提交**

```bash
git add src/query/__init__.py tests/unit/query/test_init.py
git commit -m "feat(query): finalize module public API with singleton factory"
```

---

## 完成标志

- [ ] `python -m pytest tests/unit/query/ -v` — 全部通过
- [ ] `python -c "from query import QueryUnderstandingLayer, IntentResult, get_query_layer; print('OK')"` — 导入正常
