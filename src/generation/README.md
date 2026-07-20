# 生成模块 — Prompt 组装 / 模型路由 / 生成 / 事实核查 / 引用构建

## 模块概述

生成层是 RAG 在线 Pipeline 的最终核心层，负责：

- **上下文组装**：去重（余弦/文本降级）、字符预算分配（Top-1 置顶缓解 Lost-in-the-Middle）
- **模型路由**：按意图（lookup/procedure → lightweight；concept/compare → default）选择模型和温度
- **Prompt 组装**：加载 `config/prompts/{intent}.yaml` 模板，填充 `{context}` 和 `{query}`
- **事实核查**：将回答拆解为独立断言，逐条核查是否被检索资料支撑（supported/unsupported/contradicted）
- **引用构建**：从 reranked chunks 生成 `Source` 列表

**设计哲学**：
- 每层可替换：LLM 通过构造函数注入（duck typing `ainvoke`），路由规则独立于生成逻辑
- 失败不阻塞：LLM 异常 → `answer=""`；核查异常 → 跳过核查置信度 ×0.8
- 检索不足分层处理：INSUFFICIENT 短路不调 LLM；NEED_MORE 正常生成 + 降级标注

## 文件结构

```
generation/
├── __init__.py              # 导出 + get_generation_layer 单例工厂
├── layer.py                 # GenerationLayer 主编排器
├── prompt_assembler.py      # 上下文组装（去重/预算/拼接）
├── llm_router.py            # 模型路由 + Prompt 模板加载
├── fact_checker.py          # 事实核查（断言拆解/核查/警示标注）
└── citation_builder.py      # Chunk → Source 引用映射
```

## 快速开始

```python
from generation import get_generation_layer, reset_generation_layer
from models.context import PipelineContext
from models.enums import Intent, RetrievalEval

# 初始化（首次调用时传入 LLM 客户端）
llm = create_llm_client()                      # 由上层 core 创建并注入
layer = get_generation_layer(llm)

# 构造上下文（通常由 query + retrieval 层产出的 ctx 传入）
ctx = PipelineContext(query="什么是RAG？")
ctx.intent = Intent.CONCEPT
ctx.retrieval_eval = RetrievalEval.SUFFICIENT
ctx.reranked = [chunk1, chunk2, ...]           # 来自 retrieval 层

# 生成
ctx = await layer.generate(ctx)
# → ctx.answer        LLM 生成的回答（含事实核查警示标注）
# → ctx.sources       引用来源列表
# → ctx.confidence    置信度（0.6*rerank_avg + 0.4*fact_pass_rate）
# → ctx.assembled_prompt  组装后的完整 prompt（调试用）

# 测试用重置
reset_generation_layer()
```

## 编排流程

```
ctx.reranked
    │
    ├── [INSUFFICIENT] → 短路返回（answer=""、is_fallback=True、不调 LLM）
    ├── [NEED_MORE]    → 标记 fallback_level="partial"，继续
    └── [SUFFICIENT]   → 正常流程

1. PromptAssembler.assemble()     去重 → 字符预算 → 编号拼接
2. LLMRouter.route(intent)        → RouteResult（模型名/温度/Prompt）
3. 填充 {context}/{query}         → ctx.assembled_prompt
4. LLM.ainvoke                    → raw_answer
5. FactChecker.check()            → (断言结果列表, 通过率)
6. FactChecker.inject_warnings()  → 注入警示标注
7. CitationBuilder.build()        → ctx.sources
8. _compute_confidence()           → ctx.confidence
```

## 组件 API 速查

| 组件 | 入口 | 说明 |
|------|------|------|
| `GenerationLayer` | `layer.generate(ctx)` | 主编排器，串联所有子组件 |
| `PromptAssembler` | `assembler.assemble(chunks, ...)` | 去重 + 预算 + 拼接 |
| `LLMRouter` | `router.route(intent)` | 返回 `RouteResult`（模型/温度/模板） |
| `FactChecker` | `checker.check(answer, context)` | 返回 `(list[FactCheckResult], pass_rate)` |
| `CitationBuilder` | `CitationBuilder.build(reranked)` | Chunk → `list[Source]` |

## 配置

生成模块的配置位于 `config/{env}.yaml` 的 `generation:` 节，通过 `settings.generation` 访问（参见 `src/config/settings.py` 的 `GenerationConfig`）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `dedup_threshold` | 0.85 | 上下文去重余弦阈值（0~1） |
| `max_context_chars` | 9000 | 上下文字符预算（近似 6000 tokens） |
| `fact_check_enabled` | true | 事实核查开关 |
