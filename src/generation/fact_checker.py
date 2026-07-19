"""FactChecker — 事实核查：answer 拆断言 → 逐条核查 → 警示标注注入

单次 LLM 调用完成"拆断言 + 逐条核查"（设计文档 5.7）。
任何失败（LLM 超时 / JSON 解析失败）降级为 ([], 1.0)，绝不阻塞答案返回。
"""
import json
from dataclasses import dataclass
from typing import Any

from logger import logger

_VALID_STATUSES = {"supported", "unsupported", "contradicted"}


@dataclass
class FactCheckResult:
    """单条断言核查结果"""
    claim: str
    status: str  # supported | unsupported | contradicted


class FactChecker:
    """check(answer, context) → (核查结果列表, 通过率)"""

    def __init__(self, llm: Any, temperature: float = 0):
        # TODO: 替换 Any 为 LLMProtocol，统一 llm 参数类型
        self._llm = llm
        self._temperature = temperature

    async def check(
        self, answer: str, context: str
    ) -> tuple[list[FactCheckResult], float, bool]:
        """返回 (核查结果, 通过率, 是否降级)

        通过率 = supported / total；空输入返回 ([], 1.0, False)。
        降级标志为 True 时调用方应对置信度施加额外惩罚。
        """
        if not answer.strip() or not context.strip():
            return [], 1.0, False

        prompt = self._build_prompt(answer, context)
        try:
            response = await self._llm.ainvoke(
                prompt, temperature=self._temperature
            )
            raw = response.content if hasattr(response, "content") else str(response)
            results = self._parse_response(raw)
        except Exception:
            logger.warning("FactChecker LLM 调用或解析失败，跳过核查")
            return [], 1.0, True

        if not results:
            return [], 1.0, False
        supported = sum(1 for r in results if r.status == "supported")
        return results, supported / len(results), False

    def inject_warnings(self, answer: str, results: list[FactCheckResult]) -> str:
        """将 unsupported/contradicted 断言追加为警示标注；claim 截断防溢出"""
        issues = [r for r in results if r.status != "supported"]
        if not issues:
            return answer

        lines = [answer, "", "> [警告] 以下内容请谨慎参考："]
        for r in issues:
            claim = r.claim[:200]  # 截断，防 LLM 异常输出过长
            if r.status == "contradicted":
                lines.append(f"> - 「{claim}」与参考资料冲突")
            else:
                lines.append(f"> - 「{claim}」未在参考资料中找到依据")
        return "\n".join(lines)

    def _build_prompt(self, answer: str, context: str) -> str:
        return (
            "你是一个严格的事实核查器。将「待核查回答」拆解为独立的事实断言，"
            "逐条判断是否被「参考资料」支撑。\n"
            "\n"
            "status 判定标准：\n"
            "- supported: 断言内容能在参考资料中找到直接依据\n"
            "- unsupported: 参考资料中没有相关信息\n"
            "- contradicted: 断言与参考资料内容矛盾\n"
            "\n"
            "输出严格 JSON 数组格式，不要输出任何其他内容：\n"
            '[{"claim": "断言内容", "status": "<supported|unsupported|contradicted>"}]\n'
            "\n"
            f"## 参考资料\n{context}\n"
            "\n"
            f"## 待核查回答\n{answer}"
        )

    def _parse_response(self, raw: str) -> list[FactCheckResult]:
        json_str = self._extract_json_array(raw)
        if json_str is None:
            # 不将 LLM 原始输出写入异常消息（上层 logger 有脱敏，此处纵深防御）
            logger.debug("LLM 响应中未找到 JSON 数组（前200字）: %.200s", raw)
            raise ValueError("FactChecker: LLM 响应中未找到 JSON 数组")

        data = json.loads(json_str)
        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            claim_val = item.get("claim")
            if claim_val is None or claim_val == "":
                continue
            claim = str(claim_val).strip()
            if not claim:
                continue
            status = str(item.get("status", "")).strip().lower()
            if status not in _VALID_STATUSES:
                status = "unsupported"  # 未知 status 保守处理
            results.append(FactCheckResult(claim=claim, status=status))
        return results

    @staticmethod
    def _extract_json_array(raw: str) -> str | None:
        """从 LLM 响应中提取最外层 JSON 数组（处理字符串值中的方括号）"""
        stripped = raw.strip()
        # 快速路径：整个响应就是合法 JSON 数组
        if stripped.startswith("["):
            try:
                json.loads(stripped)
                return stripped
            except json.JSONDecodeError:
                pass

        # 慢速路径：括号计数提取最外层数组
        start = raw.find("[")
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
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
        return None
