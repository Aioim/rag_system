"""parse_react_output 测试"""
from agent.react_agent import parse_react_output


class TestParseReactOutput:
    def test_parses_search_action(self):
        text = "THOUGHT: 需要搜索RAG相关资料\nACTION: search\nQUERY: RAG 检索增强生成"
        result = parse_react_output(text)
        assert result["thought"] == "需要搜索RAG相关资料"
        assert result["action"] == "search"
        assert result["query"] == "RAG 检索增强生成"

    def test_parses_web_search_action(self):
        text = "THOUGHT: 内部知识库无结果\nACTION: web_search\nQUERY: RAG architecture 2026"
        result = parse_react_output(text)
        assert result["action"] == "web_search"

    def test_parses_finish_action(self):
        text = "THOUGHT: 信息已充分，可以回答\nACTION: FINISH"
        result = parse_react_output(text)
        assert result["thought"] == "信息已充分，可以回答"
        assert result["action"] == "finish"
        assert result["query"] is None

    def test_parses_multiline_thought(self):
        text = "THOUGHT: 第一行思考\n第二行继续思考\n第三行总结\nACTION: FINISH"
        result = parse_react_output(text)
        assert "第一行思考" in result["thought"]
        assert result["action"] == "finish"

    def test_rejects_unknown_action(self):
        text = "THOUGHT: test\nACTION: unknown\nQUERY: test"
        result = parse_react_output(text)
        assert "parse_error" in result

    def test_handles_missing_thought(self):
        text = "ACTION: search\nQUERY: test"
        result = parse_react_output(text)
        assert "parse_error" in result

    def test_handles_empty_input(self):
        result = parse_react_output("")
        assert "parse_error" in result

    def test_trims_whitespace(self):
        text = "  THOUGHT:  需要搜索  \n  ACTION:  search  \n  QUERY:  hello world  "
        result = parse_react_output(text)
        assert result["thought"] == "需要搜索"
        assert result["action"] == "search"
        assert result["query"] == "hello world"

    def test_finish_ignores_query_field(self):
        text = "THOUGHT: done\nACTION: FINISH\nQUERY: should be ignored"
        result = parse_react_output(text)
        assert result["action"] == "finish"
        assert result["query"] is None
