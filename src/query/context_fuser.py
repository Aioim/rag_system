"""多轮上下文融合 — 指代消解 + 追问补全"""
from config import settings
from logger import logger
from session.manager import SessionManager


class ContextFuser:
    """将多轮对话中的追问/指代补全为独立完整问题

    SessionManager 用于获取对话历史。
    """

    def __init__(self, llm, session_manager: SessionManager, temperature: float = 0):
        self._llm = llm
        self._session_manager = session_manager
        self._temperature = temperature
        self._max_history_msgs = settings.session.max_history_rounds * 2

    async def fuse(self, query: str, session_id: str, session=None) -> str:
        if session is None:
            session = self._session_manager.get(session_id)
        if session is None or not session.messages:
            return query

        history = self._format_history(session.messages)
        prompt = self._build_prompt(history, query)
        try:
            response = (await self._llm.ainvoke(prompt, temperature=self._temperature)).content
            result = response.strip()
            return result if result else query
        except Exception:
            logger.warning("ContextFuser LLM 调用失败，返回原始 query")
            return query

    def _format_history(self, messages) -> str:
        max_msgs = self._max_history_msgs
        lines = []
        if max_msgs <= 0:
            return ""
        for msg in messages[-max_msgs:]:
            if msg.role == "user":
                role = "用户"
            elif msg.role == "system":
                role = "系统"
            else:
                role = "助手"
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


# ============================================================================
# 自测：用 Mock LLM + Mock Session 演示上下文融合
# ============================================================================
if __name__ == "__main__":
    import asyncio


    class _MockSession:
        def __init__(self, messages):
            self.messages = messages

    class _MockMsg:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    from types import SimpleNamespace

    class _MockLLM:
        async def ainvoke(self, prompt, **_kw):
            # 简单模拟：检测到指代词就补全，否则原样返回
            if "那个" in prompt.split("当前问题：")[-1] if "当前问题：" in prompt else "":
                return SimpleNamespace(content="年假的申请条件和所需材料")
            return SimpleNamespace(content=prompt.split("当前问题：")[-1].strip() if "当前问题：" in prompt else "")

    class _MockSM:
        def get(self, sid):
            if sid == "s1":
                return _MockSession([
                    _MockMsg("user", "年假怎么申请？"),
                    _MockMsg("assistant", "年假需要提前在OA系统提交申请..."),
                ])
            return None


    async def main():
        fuser = ContextFuser(_MockLLM(), _MockSM())
        print("=" * 60)
        print("ContextFuser 自测")
        print("=" * 60)

        # 无 session
        result = await fuser.fuse("五险一金缴纳比例？", "nonexistent")
        print(f"  无 session: '{result}'")

        # 有 session，指代消解
        result = await fuser.fuse("那个需要什么材料？", "s1")
        print(f"  有 session (指代消解): '{result}'")

    asyncio.run(main())
