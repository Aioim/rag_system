"""意图分类 + 清晰度判断（合并单次 LLM 调用）"""
import json
from dataclasses import dataclass

from logger import logger
from models.enums import Intent


@dataclass
class IntentResult:
    """意图分类结果"""
    intent: Intent
    is_clear: bool
    clarification_question: str | None


class IntentClassifier:
    """意图分类器 — 单次 LLM 调用完成意图分类 + 清晰度判断

    失败时降级为 intent=CONCEPT, is_clear=True。
    """

    def __init__(self, llm, temperature: float | None = None):
        self._llm = llm
        self._temperature = temperature

    async def classify(self, query: str) -> IntentResult:
        prompt = self._build_prompt(query)
        try:
            kwargs = {}
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            raw = (await self._llm.ainvoke(prompt, **kwargs)).content
            return self._parse_response(raw)
        except Exception:
            logger.warning("IntentClassifier LLM 调用或解析失败，降级为默认意图")
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
            '{"intent": "<concept|procedure|compare|lookup>", "is_clear": true/false, "clarification_question": null/"具体澄清问题"}\n'
            "\n"
            f"用户问题：{query}"
        )

    def _parse_response(self, raw: str) -> IntentResult:
        # 用括号计数提取最外层 JSON（处理字符串值中的花括号）
        json_str = self._extract_json(raw)
        if json_str is None:
            raise ValueError(f"No JSON found in response: {raw[:200]}")

        data = json.loads(json_str)

        intent_str = data.get("intent", "concept")
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.CONCEPT

        raw_clear = data.get("is_clear", True)
        # 防御：bool("false") 在 Python 中为 True，显式处理各类型
        if raw_clear is None:
            is_clear = False  # null → 问题不清晰
        elif isinstance(raw_clear, str):
            is_clear = raw_clear.strip().lower() != "false"
        elif isinstance(raw_clear, bool):
            is_clear = raw_clear
        else:
            is_clear = bool(raw_clear)
        clarification_question = data.get("clarification_question") if not is_clear else None

        return IntentResult(
            intent=intent,
            is_clear=is_clear,
            clarification_question=clarification_question,
        )

    @staticmethod
    def _extract_json(raw: str) -> str | None:
        """用括号计数提取最外层 JSON 对象，追踪字符串边界避免误截断"""
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
        return None


# ============================================================================
# 自测：用 Mock LLM 演示意图分类 + 清晰度判断
# ============================================================================
if __name__ == "__main__":
    import asyncio


    from types import SimpleNamespace

    class _MockLLM:
        """模拟 LLM — 根据 query 关键词返回不同分类结果"""

        async def ainvoke(self, prompt, **_kw):
            query = prompt.split("用户问题：")[-1].strip() if "用户问题：" in prompt else ""
            if any(w in query for w in ("如何", "怎么", "步骤", "申请")):
                return SimpleNamespace(content='{"intent": "procedure", "is_clear": true, "clarification_question": null}')
            if any(w in query for w in ("区别", "对比", "哪个好")):
                return SimpleNamespace(content='{"intent": "compare", "is_clear": true, "clarification_question": null}')
            if any(w in query for w in ("帮帮我", "救救我")):
                return SimpleNamespace(content='{"intent": "concept", "is_clear": false, "clarification_question": "您想了解什么内容？"}')
            return SimpleNamespace(content='{"intent": "concept", "is_clear": true, "clarification_question": null}')


    async def main():
        classifier = IntentClassifier(_MockLLM())
        queries = [
            "什么是RAG？",
            "如何申请年假？",
            "区别TCP和UDP",
            "帮帮我",
        ]
        print("=" * 60)
        print("IntentClassifier 自测")
        print("=" * 60)
        for q in queries:
            r = await classifier.classify(q)
            clear = "清晰" if r.is_clear else "模糊"
            print(f"  Query: {q}")
            print(f"    → intent={r.intent.value}, is_clear={clear}")
            if r.clarification_question:
                print(f"    → 澄清追问: {r.clarification_question}")
            print()

    asyncio.run(main())
