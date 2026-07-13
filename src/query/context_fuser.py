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
