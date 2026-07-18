# Generation 模块实现计划

> 日期：2026-07-18 · 分支：master（用户已确认） · 方法：TDD（先测试后实现）

## Context

第 2 期开发中，retrieval 已完成，`src/generation/` 是 RAG 在线 Pipeline 最后一个核心层（兜底模块除外）。设计规范见 `docs/superpowers/specs/2026-07-09-rag-enterprise-qa-design.md` 第 5.5–5.7 节：

- **5.5 上下文组装**：reranked chunks → 去重（余弦>0.85 保留高分）→ Token 预算 → Lost-in-the-Middle 缓解（Top-1 置顶）→ assembled prompt
- **5.6 生成层**：模型路由 + 按 intent 温度控制 + 按 intent 选 Prompt 模板
- **5.7 事实核查（一期）**：answer 拆断言 → 逐条核查（supported/unsupported/contradicted）→ 注入警示标注

**用户已确认的范围决策**：
1. ✅ 包含 FactChecker（事实核查）
2. ❌ 流式输出本期不实现（接口预留扩展点）
3. ✅ 模型路由用**纯规则**：lookup/procedure → lightweight（haiku），concept/compare → default（sonnet）
4. ✅ 直接在 master 开发；**先提交现有 retrieval 未提交修改**再开始

**已验证的基础设施**（可直接复用）：
- `PipelineContext` 已预留字段：`assembled_prompt` / `answer` / `sources` / `confidence` / `fallback_level` / `is_fallback`（src/models/context.py）
- `Source(doc_id, doc_title, chunk_text, score)`（src/models/api.py）
- `Intent` / `RetrievalEval` 枚举（src/models/enums.py）
- `LLMConfig`：default="claude-sonnet-5"、lightweight="claude-haiku-4-5"、`temperatures` 按 intent（src/config/settings.py）
- Prompt 模板已就绪：`config/prompts/{concept,procedure,compare,lookup}.yaml`（system + user_template，`{context}`/`{query}` 占位符）
- LLM 依赖注入模式（duck typing：`async ainvoke(prompt, **kwargs) → obj.content`），不直接实例化客户端
- 单例模式参考 `src/query/__init__.py`（双重检查锁 + id 校验警告）

**已验证的关键事实**：
- `chunk.metadata` 目前**不含** `doc_title`（ingestion/chunker.py 创建 chunk 时 metadata 为空）；docstore 持久化会保留 metadata（store.py:107）→ 需要在 chunker 补一行写入，否则 Source.doc_title 永远是 UUID
- `pyproject.toml` known-first-party 目前无 "generation"（第 114 行）
- pytest：asyncio_mode="auto"，pythonpath=["src"]

## 设计决策

| 决策点 | 结论 |
|--------|------|
| 文件拆分 | 5 个源文件：prompt_assembler / llm_router / fact_checker / citation_builder / layer；设计文档中的 generator.py 职责合并进 layer.py，避免空壳编排器 |
| 上下文组装归属 | 放 generation/prompt_assembler.py（消费方就是生成层）；embedding 为 None 时降级为文本精确去重 |
| Token 估算 | 不引入 tokenizer，字符数近似（中文场景），独立纯函数 `_estimate_tokens`，后续可替换 |
| doc_title 来源 | `chunk.metadata.get("doc_title", chunk.doc_id)` 降级写法 + 在 ingestion chunker 补写 `doc_title`（上游一行改动，让 Source 可读，理由：generation 功能正确所需的最小上游改动） |
| confidence | `0.6 * avg_rerank_score + 0.4 * fact_check_pass_rate`；核查跳过时 pass_rate=1.0；核查异常时 confidence *= 0.8 |
| 新增 GenerationConfig | 是（dedup_threshold=0.85 / max_context_chars=9000 / fact_check_enabled=true）+ defaults.yaml `generation:` 节 |
| 事实核查模型 | 概念上路由到 lightweight（temperature=0）；失败降级：跳过核查不阻塞答案 |
| RetrievalEval 行为 | SUFFICIENT → 正常流程；NEED_MORE → 正常生成 + `fallback_level="partial"`；INSUFFICIENT → 短路不调 LLM，`answer=""`、`is_fallback=True`、`fallback_level="no_answer"`，兜底交给上层（未实现的 fallback 模块） |
| 流式 | 不实现；`_call_llm` 单独抽出为未来 stream 扩展点 |

## 任务清单

### Task 0：提交现有 retrieval 未提交修改
```bash
git add -A && git status   # 确认（含删除 src.zip）
git commit  # fix(retrieval): <按实际 diff 内容拟定消息>
```
验证：`git status` 干净；`pytest tests/ -x -q` 全绿（回归基线）。

### Task 1：GenerationConfig 配置模型
- 修改 `src/config/settings.py`：新增 `GenerationConfig(_BaseConfig)`（dedup_threshold=0.85, max_context_chars=9000, fact_check_enabled=True）+ `RAGAppConfig.generation` 字段 + `__all__` 导出
- 修改 `config/defaults.yaml`：新增 `generation:` 节
- 新建 `tests/unit/generation/__init__.py`（空）+ `test_config.py`（默认值 + 环境变量覆盖 `GENERATION__DEDUP_THRESHOLD`）
- 验证：`pytest tests/unit/generation/test_config.py -v`（先 FAIL 后 PASS）

### Task 2：ingestion chunker 写入 doc_title（上游一行改动）
- 修改 `src/ingestion/chunker.py`：创建 chunk 时 `metadata={"doc_title": <document.title>}`（具体注入点看 chunker run 如何传 doc 信息）
- 在 `tests/unit/ingestion/` 对应测试文件加断言
- 验证：`pytest tests/unit/ingestion/ -q`

### Task 3：CitationBuilder（~60 行）
- 新建 `src/generation/citation_builder.py`：`build(reranked: list[Chunk]) -> list[Source]`
- doc_title 用 `metadata.get("doc_title", doc_id)` 降级；顺序与 chunks 一致；空列表安全
- 测试 `test_citation_builder.py`：正常映射 / doc_title 降级 / 空列表
- 验证：`pytest tests/unit/generation/test_citation_builder.py -v`

### Task 4：PromptAssembler（~160 行）
- 新建 `src/generation/prompt_assembler.py`：
  - `dedup(chunks, threshold)`：余弦>阈值保留高分；embedding 缺失降级文本精确去重
  - `allocate_budget(chunks, max_chars)`：Top-1 置顶 + 字符预算截断
  - `assemble(chunks) -> str`：dedup → budget → 编号拼接（`[1] 文本...`，与模板 `{context}` 对接）
- 测试：去重（含降级）/ 预算截断 / Top-1 置顶 / 空 chunks / 单 chunk
- 验证：`pytest tests/unit/generation/test_prompt_assembler.py -v`

### Task 5：LLMRouter（~130 行）
- 新建 `src/generation/llm_router.py`：
  - `RouteResult(model_tier, model_name, temperature, system_prompt, user_template)` dataclass
  - `route(intent) -> RouteResult`：lookup/procedure → lightweight；concept/compare → default；temperature 从 `settings.llm.temperatures` 取；模板从 `config/prompts/{intent}.yaml` 懒加载缓存
  - intent 为 None/无效 → 降级 CONCEPT；YAML 缺失 → 友好错误
- 测试：4 种 intent 路由 + 温度 / 模板加载 / None 降级
- 验证：`pytest tests/unit/generation/test_llm_router.py -v`

### Task 6：FactChecker（~200 行）
- 新建 `src/generation/fact_checker.py`：三段式（`_build_prompt` / `ainvoke` / `_parse_response`），temperature=0
  - `check(answer, context) -> (list[FactCheckResult], pass_rate)`：一次 LLM 调用完成"拆断言+逐条核查"，返回 JSON 解析
  - `inject_warnings(answer, results) -> str`：unsupported/contradicted 追加警示标注
  - 降级：LLM 失败 / JSON 解析失败 / 空 answer → `([], 1.0)` 不阻塞
- 测试（MockLLM）：全 supported / 部分 unsupported / contradicted / LLM 失败降级 / 解析失败降级 / inject_warnings 文本
- 验证：`pytest tests/unit/generation/test_fact_checker.py -v`

### Task 7：GenerationLayer 主编排器（~180 行）
- 新建 `tests/unit/generation/conftest.py`：MockLLM（仿 tests/unit/query/conftest.py）+ `make_chunk` + `sample_ctx` fixture
- 新建 `src/generation/layer.py`：`GenerationLayer(llm)`，`async generate(ctx) -> PipelineContext`：
  1. INSUFFICIENT 短路（不调 LLM）；NEED_MORE 标记 partial
  2. assembler.assemble → context 文本
  3. router.route(ctx.intent) → 模板/温度/模型
  4. 填充 `{context}`/`{query}` → `ctx.assembled_prompt`
  5. `_call_llm`（异常降级 answer=""，不上抛；未来 stream 扩展点）
  6. fact_check（enabled 时）→ inject_warnings
  7. citation_builder.build → `ctx.sources`
  8. `_compute_confidence` → `ctx.confidence`；metadata 记录耗时
- 测试：SUFFICIENT 完整流 / NEED_MORE partial / INSUFFICIENT 短路（LLM calls==0）/ LLM 失败降级 / 核查失败置信度 *0.8 / confidence 公式 / ctx 各字段写入
- 验证：`pytest tests/unit/generation/test_layer.py -v`

### Task 8：__init__.py 工厂 + 单例（~50 行）
- 新建 `src/generation/__init__.py`：`get_generation_layer(llm)` 双重检查锁单例（仿 query/__init__.py，含 id 校验警告）+ `reset_generation_layer()` + `__all__`
- 测试 `test_init.py`：单例 / reset / 导出
- 验证：`pytest tests/unit/generation/test_init.py -v`

### Task 9：文档同步 + 收尾
- 新建 `src/generation/README.md`（仿 src/config/README.md 结构：概述/文件结构/快速开始/组件说明/配置表）
- 新建 `docs/superpowers/plans/2026-07-18-generation-module.md`（本计划存档）
- 修改 `CLAUDE.md`：generation 标记 ✅ + 结构树子文件 + 「当前开发阶段」+ 开发要点新增「生成模块」用法节
- 修改 `pyproject.toml`：known-first-party 追加 `"generation"`
- 全量验证（见下）

### Task 10：代码审查 + 提交
- 按 requesting-code-review 流程用 code-reviewer / security-reviewer 审查（fact_checker 有 LLM 输出解析，注意注入面）
- 修复 CRITICAL/HIGH 问题
- 提交：`feat(generation): add generation layer with prompt assembly, routing, fact check`

## 验证

```bash
# 模块级（含覆盖率 ≥80%）
pytest tests/unit/generation/ -v --cov=src/generation --cov-report=term-missing
# 全量回归
pytest tests/ -x -q
# 风格
ruff check src/generation/ tests/unit/generation/ src/config/settings.py src/ingestion/chunker.py
```

## 依赖顺序

Task 0 → Task 1、2 可并行 → Task 3/4/5/6 可并行（依赖 1）→ Task 7（依赖 3-6）→ Task 8 → Task 9 → Task 10
