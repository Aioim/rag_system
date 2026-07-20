# Agent 模块 — ReAct 代理模式

基于 ReAct（Reasoning + Acting）范式的知识库问答代理，使 LLM 自主决定检索时机和内容。

## 架构

```
用户 Query → ReActAgent 循环 (Thought→Action→Observe) → 生成答案
                   │
           ┌───────┴───────┐
      SearchTool      WebSearchTool
      (向量+BM25)      (DuckDuckGo)
```

## 快速使用

```python
from agent import get_react_agent, reset_react_agent

agent = get_react_agent(llm)

# 非流式
result = await agent.run("什么是RAG？", collection="default")
print(result.total_iterations)  # 循环轮次
print(result.react_traces)      # 推理链

# 流式
async for event in agent.run_stream("RAG优化方向", "default"):
    print(event.event, event.data)
```

## 通过 RAGPipeline 使用

```python
from core import get_rag_pipeline

pipeline = get_rag_pipeline(llm, session_manager)

# ReAct 模式
ctx = await pipeline.run("什么是RAG？", mode="react")

# 返回推理链
ctx = await pipeline.run("复杂问题", mode="react", show_reasoning=True)
print(ctx.metadata["react_traces"])
```

## 核心组件

| 组件 | 职责 |
|------|------|
| `ReActAgent` | 思考→行动→观察 主循环，LLM 决策 + 工具调用编排 |
| `SearchTool` | 封装 RetrievalLayer，每次 search 走完整混合检索 |
| `WebSearchTool` | 封装 WebSearcher，联网搜索兜底 |
| `parse_react_output` | 从 LLM 响应中提取 THOUGHT/ACTION/QUERY |

## 停止条件

| 条件 | 行为 |
|------|------|
| LLM 返回 `ACTION: FINISH` | 正常退出，进入生成 |
| 达到 `max_iterations` | 强制退出 |
| 连续重复 Action+Query | 死循环检测，退出 |
| LLM 调用异常 | 降级到 linear 模式 |

## 配置

```yaml
# config/{env}.yaml
agent:
  max_iterations: 5
  search_top_k: 3
  max_observation_chars: 3000
  llm_temperature: 0.0
  max_consecutive_duplicates: 2
```

## 测试

```bash
pytest tests/unit/agent/ -v
```
