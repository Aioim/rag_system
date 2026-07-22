# Examples 目录功能拆分重构设计

**日期**: 2026-07-22
**状态**: 已确认

## 目标

将 `examples/` 下每个模块的单一 `demo_xxx.py` 按功能粗粒度拆分为 2~4 个独立可运行的演示文件。

## 当前状态

每个模块目录仅一个文件（如 `07_query/demo_query.py`），内含多个 `banner()` 分隔的独立演示段落（200-270 行）。所有段落挤在同一文件，不便于按需运行和理解。

## 拆分原则

1. **粗粒度合并** — 相关功能合并到同一文件，每个模块 2~4 个文件
2. **独立可运行** — 每个文件包含完整的 `if __name__ == "__main__"` 入口
3. **真实 LLM** — 不再使用 MockLLM，通过项目配置使用真实 DeepSeek API
4. **自包含** — `banner()`、LLM 初始化等在每个文件内独立实现
5. **编号前缀** — 保持文件的可发现性和推荐阅读顺序
6. **保留 README.md** — 更新目录和运行说明

## 共享约定

每个示例文件遵循统一模板：

```python
"""
<文件名>.py — <一句话描述>

演示内容：
  1. ...
  2. ...

运行方式：
  cd rag0709
  python examples/<编号>_<模块>/<编号>_<名称>.py
"""

import asyncio  # (如需要)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main():  # 或 def main():
    ...


if __name__ == "__main__":
    asyncio.run(main())  # 或 main()
```

LLM 初始化约定（需要 LLM 的模块）— 使用共享 helper `examples/_llm.py`：

```python
# examples/_llm.py — 所有示例共享的 LLM 工厂
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

def create_llm(temperature: float = 0, model: str | None = None):
    """创建 ChatOpenAI 实例（DeepSeek API，OpenAI 兼容）"""
    from langchain_openai import ChatOpenAI

    api_key = settings.llm.api_key.get_secret_value()
    if not api_key:
        raise RuntimeError(
            f"未设置 {settings.llm.api_key_env} 环境变量！"
            f"请在 .env 文件中设置 LLM_API_KEY=sk-xxx"
        )

    return ChatOpenAI(
        model=model or settings.llm.default,
        base_url=settings.llm.api_base_url or "https://api.deepseek.com/v1",
        api_key=api_key,
        temperature=temperature,
    )
```

各示例文件导入：
```python
from _llm import create_llm
llm = create_llm(temperature=0)
```

## 文件拆分方案

### 01_config (配置管理) — 无外部依赖

| 文件 | 内容 |
|------|------|
| `01_basic_access.py` | PROJECT_ROOT、属性访问、点号路径、get 方法、各配置段概览、敏感信息保护 |
| `02_overrides_env.py` | CLI 参数覆盖（apply_overrides）、热重载（reload）、环境变量注入白名单、Pydantic 模型验证 |

### 02_security (安全管理) — 无外部依赖

| 文件 | 内容 |
|------|------|
| `01_secret_crypto.py` | SecretStr 防泄露容器、Fernet 加密/解密、密钥文件生成、特殊字符边界情况 |
| `02_secrets_storage.py` | SecretsManager 内存加密存储（set/get/list）、环境变量 ENC[...] 解密流程、SecureEnvLoader |

### 03_logger (日志系统) — 无外部依赖

| 文件 | 内容 |
|------|------|
| `01_basic_logging.py` | 基础日志输出（logger/security_logger）、日志级别过滤、异常记录（log_exception）、安全事件（log_security_event） |
| `02_performance.py` | @log_performance 性能监控、@log_step 步骤追踪、log_duration 上下文管理器 |
| `03_data_masking.py` | mask_sensitive_data 脱敏、MaskingEngine 脱敏引擎、自定义规则示例 |

### 04_models (数据模型) — 无外部依赖

| 文件 | 内容 |
|------|------|
| `01_enums_context.py` | Intent/RetrievalEval/FallbackLevel/DocumentStatus 枚举 + PipelineContext 核心容器 |
| `02_chunk_document.py` | Chunk 分块模型 + Document 文档模型 |
| `03_session_api.py` | Session/Message 会话模型 + ChatRequest/ChatResponse/SearchRequest/Source API 模型 + extract_json_container |

### 05_session (会话管理) — SQLite

| 文件 | 内容 |
|------|------|
| `01_crud.py` | SessionManager 创建/获取/删除会话、添加/读取消息、对话历史查询 |
| `02_context_lifecycle.py` | 上下文窗口获取（get_context）、话题切换检测、上下文压缩、TTL 过期清理 |

### 06_model (模型管理) — 只读

| 文件 | 内容 |
|------|------|
| `01_status_download.py` | 下载状态查询（status）、已下载列表（list_downloaded）、路径查询（get_path）、下载配置与策略 |
| `02_finetune.py` | 微调模型管理（list_finetuned/get_finetuned_path/remove_finetuned）、微调配置预览、训练数据格式 |

### 07_query (查询理解) — 需要 LLM

| 文件 | 内容 |
|------|------|
| `01_alias_intent.py` | 别名映射（resolve_alias/resolve_aliases_in_text/alias_manager）+ 上下文消歧 + 意图分类（IntentClassifier）+ 清晰度判断 |
| `02_fusion_rewrite.py` | 多轮上下文融合（ContextFuser：指代消解 + 追问补全）+ 查询改写（QueryRewriter：HyDE/Keyword/Synonym 并行） |
| `03_full_pipeline.py` | QueryUnderstandingLayer 完整流程 + 模糊问题短路 + 温度约定总结 |

### 08_retrieval (检索层) — 部分需要 FAISS 索引

| 文件 | 内容 |
|------|------|
| `01_architecture.py` | RetrievalLayer 架构概览、检索配置详解、RRF/MMR 公式、Self-RAG 评估结果、组件就绪状态检查 |
| `02_retrieve_demo.py` | 实际检索链路（需 FAISS 索引 + Embedding 模型）：多查询检索、粗召回/精排/评估结果展示 |

### 09_generation (生成层) — 需要 LLM

| 文件 | 内容 |
|------|------|
| `01_assembly_route.py` | PromptAssembler（上下文去重/截断/拼接）+ LLMRouter（意图路由 + 温度选取） |
| `02_generate_factcheck.py` | GenerationLayer 完整生成（SUFFICIENT/NEED_MORE/INSUFFICIENT 场景）+ FactChecker + CitationBuilder + 置信度计算 |

### 10_core (RAG Pipeline) — 需要 LLM + SQLite

| 文件 | 内容 |
|------|------|
| `01_single_multi_turn.py` | RAGPipeline 初始化、单轮问答、多轮对话（含会话上下文 + 追问补全） |
| `02_edge_cases.py` | 模糊问题短路、Pipeline 异常独立降级、Fallback 触发场景、运行统计 |

### 11_ingestion (文档处理) — 需要 Embedding 模型

| 文件 | 内容 |
|------|------|
| `01_parse_chunk.py` | 依赖检查（Embedding 模型下载状态）、解析器配置（docling/pymupdf4llm/mineru/direct）、分块策略对比（semantic/fixed/hierarchical） |
| `02_pipeline_index.py` | 创建演示文档、执行完整 Pipeline（Parser→Chunker→Embedder→FAISS）、索引文件检查、增量更新验证 |

### 12_fallback (兜底处理) — 部分需要网络

| 文件 | 内容 |
|------|------|
| `01_web_search.py` | WebSearcher 联网搜索（真实搜索测试）、搜索提供商配置、超时与错误处理 |
| `02_fallback_flow.py` | SupplementaryRetriever 补充检索、FallbackHandler 三级兜底架构、NEED_MORE→PARTIAL 流程、INSUFFICIENT→WEB_SEARCH/NO_ANSWER 流程 |

### 13_agent (ReAct Agent) — 需要 LLM

| 文件 | 内容 |
|------|------|
| `01_tools.py` | ToolResult 数据结构、SearchTool（知识库搜索）、WebSearchTool（联网搜索） |
| `02_react_loop.py` | parse_react_output 解析器、ReActAgent 完整思考循环、AgentResult 结构、SSEEvent 流式事件、重复检测机制 |

## 不影响的部分

- 各模块的 `src/` 生产代码不变
- `docs/` 设计文档不变
- `tests/` 测试目录不变

## 待更新

- `examples/README.md` — 更新目录表、运行说明、拆分后文件列表
- `CLAUDE.md` — 无需更新（examples 是演示层，不影响核心文档）
