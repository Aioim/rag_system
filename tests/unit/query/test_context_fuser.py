"""ContextFuser 测试"""
import pytest
from query.context_fuser import ContextFuser
from tests.unit.query.conftest import MockLLM


class TestContextFuser:
    @pytest.mark.asyncio
    async def test_fuse_returns_completed_query(self, session_manager):
        """将指代问题补全为完整问题"""
        llm = MockLLM(response="申请年假需要什么材料？")
        fuser = ContextFuser(llm)

        # 准备会话历史
        session_manager.get_or_create("s1")
        session_manager.add_message("s1", "user", "年假怎么申请？")
        session_manager.add_message("s1", "assistant", "年假申请需要登录OA系统...")

        session = session_manager.get("s1")
        result = await fuser.fuse("需要什么材料？", session)
        assert result == "申请年假需要什么材料？"

    @pytest.mark.asyncio
    async def test_fuse_preserves_complete_query(self, session_manager):
        """已是完整问题的，原样返回"""
        llm = MockLLM(response="五险一金缴纳比例是多少？")
        fuser = ContextFuser(llm)

        session_manager.get_or_create("s1")
        session = session_manager.get("s1")
        result = await fuser.fuse("五险一金缴纳比例是多少？", session)
        assert result == "五险一金缴纳比例是多少？"

    @pytest.mark.asyncio
    async def test_fuse_handles_nonexistent_session(self, session_manager):
        """会话不存在时返回原始 query"""
        llm = MockLLM()
        fuser = ContextFuser(llm)
        result = await fuser.fuse("任意问题", None)
        assert result == "任意问题"

    @pytest.mark.asyncio
    async def test_fuse_handles_llm_error(self, session_manager):
        """LLM 失败时降级返回原始 query"""

        class FailingLLM:
            async def ainvoke(self, prompt, **kwargs):
                raise RuntimeError("timeout")

        fuser = ContextFuser(FailingLLM())
        session_manager.get_or_create("s1")
        session = session_manager.get("s1")
        result = await fuser.fuse("需要什么材料？", session)
        assert result == "需要什么材料？"

    @pytest.mark.asyncio
    async def test_fuse_includes_history_in_prompt(self, session_manager):
        """验证 prompt 包含历史消息"""
        llm = MockLLM(response="完整问题")
        fuser = ContextFuser(llm)

        session_manager.get_or_create("s2")
        session_manager.add_message("s2", "user", "VPN怎么连接？")
        session_manager.add_message("s2", "assistant", "请下载VPN客户端...")

        session = session_manager.get("s2")
        await fuser.fuse("它的密码怎么改？", session)
        prompt = llm.calls[0][0]
        assert "VPN怎么连接" in prompt
        assert "它的密码怎么改" in prompt
