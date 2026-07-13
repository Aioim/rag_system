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
