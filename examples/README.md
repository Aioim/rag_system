# RAG 企业级知识库问答系统 — 演示示例

本目录包含每个核心模块的可运行演示脚本，按功能拆分为多个独立文件，帮助你快速验证各模块功能。

## 目录

| 目录 | 模块 | 文件 | 说明 | 需要外部依赖 |
|------|------|------|------|-------------|
| `01_config/` | 配置管理 | `01_basic_access.py` | 属性访问 / 点号路径 / 配置段概览 | 无 |
| | | `02_overrides_env.py` | CLI 覆盖 / 热重载 / 环境变量注入 | 无 |
| `02_security/` | 安全管理 | `01_secret_crypto.py` | SecretStr / Fernet 加解密 / 密钥生成 | 无 |
| | | `02_secrets_storage.py` | SecretsManager 存储 / .env 解密 | 无 |
| `03_logger/` | 日志系统 | `01_basic_logging.py` | 基础日志 / 级别过滤 / 异常/安全事件 | 无 |
| | | `02_performance.py` | @log_performance / @log_step / log_duration | 无 |
| | | `03_data_masking.py` | 敏感数据脱敏 / MaskingEngine | 无 |
| `04_models/` | 数据模型 | `01_enums_context.py` | 枚举类型 / PipelineContext | 无 |
| | | `02_chunk_document.py` | Chunk / Document 模型 | 无 |
| | | `03_session_api.py` | Session / Message / API 模型 / JSON 提取 | 无 |
| `05_session/` | 会话管理 | `01_crud.py` | 会话/消息 CRUD / 对话历史 | 无 |
| | | `02_context_lifecycle.py` | 上下文窗口 / 话题切换 / TTL | 无 |
| `06_model/` | 模型管理 | `01_status_download.py` | 下载状态 / 路径查询 / 下载策略 | 无（只读） |
| | | `02_finetune.py` | 微调模型管理 / 配置预览 / 数据格式 | 无（只读） |
| `07_query/` | 查询理解 | `01_alias_intent.py` | 别名映射 / 消歧 / 意图分类 | LLM API |
| | | `02_fusion_rewrite.py` | 上下文融合 / 查询改写 | LLM API |
| | | `03_full_pipeline.py` | QueryUnderstandingLayer 完整流程 | LLM API |
| `08_retrieval/` | 检索层 | `01_architecture.py` | 架构概览 / 配置 / 组件就绪状态 | 无 |
| | | `02_retrieve_demo.py` | 实际检索链路 | FAISS 索引 |
| `09_generation/` | 生成层 | `01_assembly_route.py` | Prompt 组装 / LLM 路由 | 无 |
| | | `02_generate_factcheck.py` | 完整生成 / 事实核查 / 引用 / 置信度 | LLM API |
| `10_core/` | RAG Pipeline | `01_single_multi_turn.py` | 单轮问答 / 多轮对话 | LLM API |
| | | `02_edge_cases.py` | 模糊短路 / 异常降级 / Fallback 场景 | LLM API |
| `11_ingestion/` | 文档处理 | `01_markdown_parsing.py` | Markdown 解析器选择与执行 / 产物检查 | 无 |
| | | `02_pdf_parsing.py` | PDF 解析 / 图片提取（含/不含）/ 配置说明 | docling + pymupdf |
| | | `03_chunking_strategies.py` | 三种分块策略实际对比 | 无（Semantic 需模型） |
| | | `04_full_pipeline.py` | 完整 Pipeline / 索引验证 / 增量更新 / 检索验证 | Embedding 模型 |
| `12_fallback/` | 兜底处理 | `01_web_search.py` | WebSearcher 联网搜索 | 网络 |
| | | `02_fallback_flow.py` | 三级兜底架构 / 补充检索流程 | 无 |
| `13_agent/` | ReAct Agent | `01_tools.py` | ToolResult / SearchTool / WebSearchTool | 无 |
| | | `02_react_loop.py` | ReAct 循环 / AgentResult / SSE / 重复检测 | LLM API |

## 运行方式

```bash
# 在项目根目录下运行（确保已激活虚拟环境）

# ── 无外部依赖，可直接运行 ──
python examples/01_config/01_basic_access.py
python examples/01_config/02_overrides_env.py
python examples/02_security/01_secret_crypto.py
python examples/02_security/02_secrets_storage.py
python examples/03_logger/01_basic_logging.py
python examples/03_logger/02_performance.py
python examples/03_logger/03_data_masking.py
python examples/04_models/01_enums_context.py
python examples/04_models/02_chunk_document.py
python examples/04_models/03_session_api.py
python examples/05_session/01_crud.py
python examples/05_session/02_context_lifecycle.py
python examples/06_model/01_status_download.py
python examples/06_model/02_finetune.py
python examples/08_retrieval/01_architecture.py
python examples/09_generation/01_assembly_route.py
python examples/11_ingestion/01_markdown_parsing.py
python examples/11_ingestion/02_pdf_parsing.py
python examples/11_ingestion/03_chunking_strategies.py
python examples/12_fallback/02_fallback_flow.py
python examples/13_agent/01_tools.py

# ── 需要 LLM API Key（在 .env 中配置 LLM_API_KEY=sk-xxx） ──
python examples/07_query/01_alias_intent.py
python examples/07_query/02_fusion_rewrite.py
python examples/07_query/03_full_pipeline.py
python examples/09_generation/02_generate_factcheck.py
python examples/10_core/01_single_multi_turn.py
python examples/10_core/02_edge_cases.py
python examples/13_agent/02_react_loop.py

# ── 需要 Embedding 模型（需先下载: models.download('embedding')） ──
python examples/11_ingestion/04_full_pipeline.py

# ── 需要 FAISS 索引（需先运行 ingestion 建库） ──
python examples/08_retrieval/02_retrieve_demo.py

# ── 需要网络连接（联网搜索测试） ──
python examples/12_fallback/01_web_search.py
```

## LLM 配置

需要 LLM API 的示例使用共享工厂 `examples/_llm.py`，通过 DeepSeek API（OpenAI 兼容协议）调用真实模型。

在项目根目录 `.env` 中配置：

```bash
LLM_API_KEY=sk-your-deepseek-api-key
```

支持 Fernet 加密格式：

```bash
LLM_API_KEY=ENC[base64_ciphertext]
# 加密工具: python -m security.env_encrypt encrypt <value>
```

## 设计原则

- **功能拆分**：每个文件聚焦单一功能组，便于按需运行和理解
- **自包含**：每个文件独立可运行，不依赖其他示例文件
- **真实 LLM**：使用真实 DeepSeek API，输出有实际参考价值
- **边界覆盖**：展示正常流程 + 常见边界情况
- **输出清晰**：每个步骤都有分隔线和说明文字
