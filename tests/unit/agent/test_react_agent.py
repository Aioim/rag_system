"""ReActAgent 核心循环单元测试"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.tools import SearchTool, WebSearchTool, ToolResult


def _make_search_tool(chunks_per_call: list[list] | None = None):
    """创建可编程 SearchTool mock"""
    if chunks_per_call is None:
        chunks_per_call = [[]]
    tool = MagicMock(spec=SearchTool)
    call_results = []
    for chunks in chunks_per_call:
        content = "\n".join(f"[来源: {c['doc_id']}] {c['text']}" for c in chunks)
        call_results.append(
            ToolResult(tool="search", query="", content=content,
                       chunk_count=len(chunks), elapsed_ms=100.0)
        )
    tool.run = AsyncMock(side_effect=call_results)
    return tool


def _make_web_search_tool(results: list[str] | None = None):
    if results is None:
        results = [""]
    tool = MagicMock(spec=WebSearchTool)
    call_results = [
        ToolResult(tool="web_search", query="", content=r or "",
                   chunk_count=0, elapsed_ms=200.0)
        for r in results
    ]
    tool.run = AsyncMock(side_effect=call_results)
    return tool


class TestReActAgent:
    """ReActAgent 核心循环测试"""

    @pytest.mark.asyncio
    async def test_single_round_finish(self, programmable_llm, agent_config):
        """单轮 FINISH：Agent 直接结束"""
        llm = programmable_llm([
            "THOUGHT: 这是一个简单问题，无需搜索\nACTION: FINISH"
        ])
        search = _make_search_tool()
        web = _make_web_search_tool()
        from agent.react_agent import ReActAgent
        agent = ReActAgent(llm, search, web, agent_config)

        result = await agent.run("你好", "default")

        assert result.total_iterations == 1
        assert result.react_traces[0].action == "finish"
        assert search.run.call_count == 0

    @pytest.mark.asyncio
    async def test_two_round_search_then_finish(self, programmable_llm, agent_config):
        """两轮检索后 FINISH"""
        from models.chunk import Chunk

        # chunk 用于验证数据模型兼容性
        chunk = Chunk(chunk_id="c1", doc_id="d1", text="RAG是检索增强生成",
                       chunk_index=0, embedding=None)
        chunk.rerank_score = 0.9

        llm = programmable_llm([
            "THOUGHT: 需要搜索RAG资料\nACTION: search\nQUERY: RAG架构",
            "THOUGHT: 信息充分，可以回答\nACTION: FINISH",
        ])
        search = _make_search_tool([[{"doc_id": "d1", "text": "RAG是检索增强生成"}]])
        web = _make_web_search_tool()
        from agent.react_agent import ReActAgent
        agent = ReActAgent(llm, search, web, agent_config)

        result = await agent.run("什么是RAG？", "default")

        assert result.total_iterations == 2
        assert result.react_traces[0].action == "search"
        assert result.react_traces[1].action == "finish"
        # reranked 由 RAGPipeline 统一处理，此处返回空列表
        assert len(result.reranked) == 0

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, programmable_llm, agent_config):
        """达到 max_iterations 后强制退出"""
        agent_config.max_iterations = 3
        llm = programmable_llm([
            "THOUGHT: 搜索第一次\nACTION: search\nQUERY: test1",
            "THOUGHT: 搜索第二次\nACTION: search\nQUERY: test2",
            "THOUGHT: 搜索第三次\nACTION: search\nQUERY: test3",
        ])
        search = _make_search_tool([
            [{"doc_id": "d1", "text": "result1"}],
            [{"doc_id": "d2", "text": "result2"}],
            [{"doc_id": "d3", "text": "result3"}],
        ])
        web = _make_web_search_tool()
        from agent.react_agent import ReActAgent
        agent = ReActAgent(llm, search, web, agent_config)

        result = await agent.run("test", "default")

        assert result.total_iterations == 3
        # 最后一轮 action 被强制改为 finish
        assert result.react_traces[-1].action == "finish"

    @pytest.mark.asyncio
    async def test_consecutive_duplicate_detection(self, programmable_llm, agent_config):
        """连续两轮相同 ACTION+QUERY 触发死循环检测"""
        agent_config.max_consecutive_duplicates = 2
        llm = programmable_llm([
            "THOUGHT: 搜索\nACTION: search\nQUERY: same query",
            "THOUGHT: 再搜一次\nACTION: search\nQUERY: same query",
            "THOUGHT: 还搜\nACTION: search\nQUERY: same query",
        ])
        search = _make_search_tool([
            [{"doc_id": "d1", "text": "r1"}],
            [{"doc_id": "d1", "text": "r1"}],
            [{"doc_id": "d1", "text": "r1"}],
        ])
        web = _make_web_search_tool()
        from agent.react_agent import ReActAgent
        agent = ReActAgent(llm, search, web, agent_config)

        result = await agent.run("test", "default")

        # 第2轮检测到重复，第3轮不应执行
        assert result.total_iterations <= 2

    @pytest.mark.asyncio
    async def test_web_search_when_kb_empty(self, programmable_llm, agent_config):
        """search 返回空结果后 Agent 选择 web_search"""
        llm = programmable_llm([
            "THOUGHT: 内部搜索无结果，尝试联网\nACTION: web_search\nQUERY: latest RAG trends",
            "THOUGHT: 联网搜索获得结果\nACTION: FINISH",
        ])
        search = _make_search_tool([[]])  # 空结果
        web = _make_web_search_tool(["联网搜索结果: RAG最新趋势..."])
        from agent.react_agent import ReActAgent
        agent = ReActAgent(llm, search, web, agent_config)

        result = await agent.run("RAG最新趋势", "default")

        assert result.react_traces[0].action == "web_search"
        assert result.react_traces[1].action == "finish"

    @pytest.mark.asyncio
    async def test_llm_format_error_force_finish(self, programmable_llm, agent_config):
        """LLM 输出格式异常时直接结束，不重试"""
        llm = programmable_llm([
            "随便写的一段话，没有格式",                              # 解析失败
            "THOUGHT: 正确格式\nACTION: FINISH",                    # 正常结束
        ])
        search = _make_search_tool()
        web = _make_web_search_tool()
        from agent.react_agent import ReActAgent
        agent = ReActAgent(llm, search, web, agent_config)

        result = await agent.run("test", "default")

        assert result.total_iterations == 1
        assert result.react_traces[0].action == "finish"  # 解析失败退化为 FINISH
        assert "格式" in result.react_traces[0].thought


class TestReActAgentStream:
    """ReActAgent 流式输出测试"""

    @pytest.mark.asyncio
    async def test_stream_emits_all_events(self, programmable_llm, agent_config):
        """流式模式推送完整的 react_start → thought → action → observation → react_end 事件"""
        llm = programmable_llm([
            "THOUGHT: 需要搜索\nACTION: search\nQUERY: RAG架构",
            "THOUGHT: 信息充分\nACTION: FINISH",
        ])
        search = _make_search_tool([
            [{"doc_id": "d1", "text": "RAG是检索增强生成"}]
        ])
        web = _make_web_search_tool()
        from agent.react_agent import ReActAgent
        agent = ReActAgent(llm, search, web, agent_config)

        events = []
        async for e in agent.run_stream("什么是RAG？", "default"):
            events.append(e)

        event_names = [e.event for e in events]
        assert "react_start" in event_names
        assert "thought" in event_names
        assert "action" in event_names
        assert "observation" in event_names
        assert "react_end" in event_names

    @pytest.mark.asyncio
    async def test_stream_react_end_has_stats(self, programmable_llm, agent_config):
        """react_end 事件包含 total_iterations 和 total_elapsed_ms"""
        llm = programmable_llm([
            "THOUGHT: 直接回答\nACTION: FINISH",
        ])
        search = _make_search_tool()
        web = _make_web_search_tool()
        from agent.react_agent import ReActAgent
        agent = ReActAgent(llm, search, web, agent_config)

        events = []
        async for e in agent.run_stream("hello", "default"):
            events.append(e)

        react_end = [e for e in events if e.event == "react_end"][0]
        assert react_end.data["total_iterations"] == 1
        assert "total_elapsed_ms" in react_end.data
