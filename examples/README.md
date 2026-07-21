# RAG 企业级知识库问答系统 — 演示示例

本目录包含每个核心模块的可运行演示脚本，帮助你快速验证各模块功能。

## 目录

| 目录 | 模块 | 说明 | 需要外部依赖 |
|------|------|------|-------------|
| `01_config/` | 配置管理 | YAML 加载、环境变量覆盖、热重载 | 无 |
| `02_security/` | 安全管理 | 密钥加密存储、.env 加载、脱敏 | 无 |
| `03_logger/` | 日志系统 | 结构化日志、脱敏、性能装饰器 | 无 |
| `04_models/` | 数据模型 | PipelineContext、Chunk、枚举等 | 无 |
| `05_session/` | 会话管理 | SQLite 会话 CRUD、话题切换 | 无 |
| `06_model/` | 模型管理 | 下载状态、模型列表、微调查询 | 无（只读） |
| `07_query/` | 查询理解 | 别名映射、意图分类、查询改写 | Mock LLM |
| `08_retrieval/` | 检索层 | 向量检索、BM25、RRF 融合、Rerank | FAISS 索引 |
| `09_generation/` | 生成层 | Prompt 组装、生成、事实核查 | Mock LLM |
| `10_core/` | RAG Pipeline | 全链路编排 | Mock LLM |
| `11_ingestion/` | 文档处理 | 解析、分块、Embedding、入库 | Embedding 模型 |
| `12_fallback/` | 兜底处理 | 补充检索、联网搜索、诚实告知 | 网络（联网搜索） |
| `13_agent/` | ReAct Agent | 思考-行动-观察循环 | Mock LLM |

## 运行方式

```bash
# 在项目根目录下运行（确保已激活虚拟环境）

# 无外部依赖，可直接运行
python examples/01_config/demo_config.py
python examples/02_security/demo_security.py
python examples/03_logger/demo_logger.py
python examples/04_models/demo_models.py
python examples/05_session/demo_session.py
python examples/06_model/demo_model.py

# 需要 Mock LLM（自动 mock，无需外部服务）
python examples/07_query/demo_query.py
python examples/09_generation/demo_generation.py
python examples/10_core/demo_core.py
python examples/13_agent/demo_agent.py

# 需要 Embedding 模型（需先下载: models.download('embedding')）
python examples/11_ingestion/demo_ingestion.py

# 需要 FAISS 索引（需先运行 ingestion 建库）
python examples/08_retrieval/demo_retrieval.py

# 需要网络连接（联网搜索测试）
python examples/12_fallback/demo_fallback.py
```

## 设计原则

- **自包含**：每个示例独立可运行，不依赖其他示例
- **Mock 友好**：对 LLM / 重模型使用 Mock，核心逻辑无需外部服务
- **边界覆盖**：展示正常流程 + 常见边界情况
- **输出清晰**：每个步骤都有分隔线和说明文字
