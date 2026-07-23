# 本地 LLM 推理引擎设计

日期: 2026-07-23 | 状态: approved

## 目标

为 `model` 模块的 `inference.py` 实现 `generate()` 本地 LLM 推理能力，同时提供 OpenAI 兼容代理服务，使系统支持纯本地运行（无需云端 API）。

两种使用模式：
1. **进程内推理** — `LocalLLM` 类直接调用，`LocalLLMAdapter` 实现 `LLMProtocol` 无缝接入现有管线
2. **OpenAI 兼容代理服务** — 独立 FastAPI 进程，实现 `/v1/chat/completions` 端点

## 架构

```
src/model/
├── inference.py          ← 增强: 新增 LocalLLM + generate() + get_local_llm()
├── llm_adapter.py         ← 新增: LocalLLMAdapter 实现 LLMProtocol
├── proxy/                 ← 新增: OpenAI 兼容代理服务
│   ├── __init__.py
│   ├── server.py          ← FastAPI 子应用
│   ├── models.py          ← Pydantic 请求/响应模型
│   └── __main__.py        ← CLI 入口 python -m model.proxy
├── manager.py             ← 增强: models.generate() / models.local_llm
├── downloader.py          ← 增强: download_gguf() 下载 GGUF 文件
├── __init__.py            ← 更新导出
└── finetune/              ← 不变
```

**数据流：**
```
应用代码
  ├─ inference.generate(prompt) → str
  ├─ LocalLLMAdapter.ainvoke(prompt) → BaseMessage  ← 实现 LLMProtocol
  └─ POST /v1/chat/completions → OpenAI JSON 响应
       ↑ 内部复用 get_local_llm() 单例
```

## LocalLLM — 推理引擎

封装 llama-cpp-python，管理 GGUF 模型加载/卸载/推理。

```python
class LocalLLM:
    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_threads: int | None = None,   # null=auto
        n_gpu_layers: int = 0,           # 0=纯CPU
        verbose: bool = False,
    ): ...

    # 属性
    model_path: Path
    is_loaded: bool

    # 推理
    def __call__(self, prompt: str, **kwargs) -> str: ...
    def stream(self, prompt: str, **kwargs) -> Iterator[str]: ...
    async def ainvoke(self, prompt: str, **kwargs) -> str: ...  # run_in_executor

    # 生命周期
    def load(self) -> None: ...
    def unload(self) -> None: ...
```

**关键设计决策：**
- **懒加载** — 首次 `__call__` 才初始化 Llama 实例
- **异步包装** — `ainvoke()` 通过 `asyncio.to_thread()` 避免阻塞事件循环
- **线程安全** — `threading.Lock` 保护推理调用（Llama 实例非线程安全）
- **双检锁** — 加载时保证多线程只加载一次

**推理参数（**kwargs 透传）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_tokens | 512 | 最大生成 token 数 |
| temperature | 0.7 | 温度 |
| top_p | 0.95 | nucleus sampling |
| stop | [] | 停止词列表 |

## LocalLLMAdapter — LLMProtocol 适配器

`src/model/llm_adapter.py`，包装 `LocalLLM`，实现 `LLMProtocol`：

```python
class LocalLLMAdapter:
    def __init__(self, llm: LocalLLM, default_temperature: float = 0.0): ...
    async def ainvoke(self, prompt: str, **kwargs) -> _FakeMessage: ...
    # 返回 _FakeMessage，有 .content 属性
```

可直接注入 `RAGPipeline(llm=adapter, ...)` 或 `GenerationLayer(llm=adapter)`。

## OpenAI 兼容代理服务

`src/model/proxy/`，FastAPI 子应用：

```
GET  /v1/models              → {"data": [{"id": "local-llm", ...}]}
POST /v1/chat/completions    → OpenAI 格式请求/响应
GET  /health                 → {"status": "ok"}
```

**Pydantic 模型**（`proxy/models.py`）：`ChatCompletionRequest`, `Message`, `ChatCompletionResponse`, `Choice`, `ModelList`

**CLI 入口**：`python -m model.proxy --port 8080 --model <path>`

**LangChain 集成**：`ChatOpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")`

## 配置

`config/dev.yaml` 新增 `inference` 段：

```yaml
inference:
  llm_model: Qwen/Qwen3-0.6B
  gguf_file: Qwen3-0.6B-Q4_K_M.gguf
  n_ctx: 4096
  n_threads: null              # null=auto
  n_gpu_layers: 0              # 0=纯CPU
  default_max_tokens: 512
  default_temperature: 0.0     # 管线用（确定性）
  verbose: false
```

`LLMConfig` 新增：
```python
class LLMConfig(_BaseConfig):
    local_enabled: bool = False   # 启用本地 LLM
```

`settings.py` 新增 `InferenceConfig` Pydantic 模型。

## 依赖

`pyproject.toml` 新增可选依赖组：

```toml
[project.optional-dependencies]
local-llm = [
    "llama-cpp-python>=0.3.0",
]
```

## downloader.py — GGUF 下载

`ModelDownloader` 新增方法：

```python
def download_gguf(self, model_id: str, gguf_filename: str) -> Path:
    """下载指定 GGUF 文件到模型目录（huggingface_hub.hf_hub_download）"""
```

## 文件清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/model/inference.py` | **修改** | 新增 `LocalLLM` + `generate()` + `get_local_llm()` |
| `src/model/llm_adapter.py` | **新增** | `LocalLLMAdapter` 实现 `LLMProtocol` |
| `src/model/proxy/__init__.py` | **新增** | 导出 |
| `src/model/proxy/server.py` | **新增** | FastAPI OpenAI 兼容服务 |
| `src/model/proxy/models.py` | **新增** | Pydantic 请求/响应模型 |
| `src/model/proxy/__main__.py` | **新增** | CLI 入口 `python -m model.proxy` |
| `src/model/__init__.py` | **修改** | 导出新组件 |
| `src/model/manager.py` | **修改** | `models.generate()` / `models.local_llm` |
| `src/model/downloader.py` | **修改** | `download_gguf()` |
| `src/config/settings.py` | **修改** | 新增 `InferenceConfig`，`LLMConfig` 加 `local_enabled` |
| `config/dev.yaml` | **修改** | 新增 `inference:` 段 |
| `pyproject.toml` | **修改** | 新增 `local-llm` 可选依赖 |
| `tests/unit/model/test_inference.py` | **修改** | 新增 generate / LocalLLM 测试 |
| `tests/unit/model/test_llm_adapter.py` | **新增** | LLMProtocol 适配器测试 |
| `tests/unit/model/test_proxy.py` | **新增** | 代理服务测试 |

## 实现依赖顺序

1. `config` — `InferenceConfig` + `dev.yaml` 配置
2. `inference.py` — `LocalLLM` 类 + `generate()` + `get_local_llm()`
3. `llm_adapter.py` — `LocalLLMAdapter`
4. `downloader.py` — `download_gguf()`
5. `proxy/` — OpenAI 兼容服务
6. `manager.py` / `__init__.py` — 导出更新
7. `pyproject.toml` — 依赖更新
8. 测试
