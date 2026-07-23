# 本地 LLM 推理引擎实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 model 模块的 generate() 本地 LLM 推理能力 + OpenAI 兼容代理服务，使系统支持纯本地运行。

**Architecture:** 在现有 inference.py（已实现 encode/rerank）基础上新增 LocalLLM 类（llama-cpp-python + GGUF），LocalLLMAdapter 实现 LLMProtocol 无缝接入管线，proxy/ 子包提供 OpenAI 兼容 HTTP 服务。

**Tech Stack:** llama-cpp-python, FastAPI, Pydantic v2, asyncio, huggingface_hub

## Global Constraints

- Python >= 3.11
- pydantic >= 2.0
- 遵循项目现有模式：单例用双检锁，配置用 Pydantic BaseModel + Field
- 模型默认: Qwen/Qwen3-0.6B（可配置）
- llama-cpp-python >= 0.3.0

---

### Task 1: 配置模型 — InferenceConfig + dev.yaml + LLMConfig.local_enabled

**Files:**
- Modify: `src/config/settings.py` (add InferenceConfig, update LLMConfig)
- Modify: `config/dev.yaml` (add inference section)
- Modify: `src/config/__init__.py` (export InferenceConfig)

**Interfaces:**
- Produces: `InferenceConfig` class with fields:
  - `llm_model: str = "Qwen/Qwen3-0.6B"`
  - `gguf_file: str = "Qwen3-0.6B-Q4_K_M.gguf"`
  - `n_ctx: int = 4096`
  - `n_threads: int | None = None`
  - `n_gpu_layers: int = 0`
  - `default_max_tokens: int = 512`
  - `default_temperature: float = 0.0`
  - `verbose: bool = False`
- Produces: `LLMConfig.local_enabled: bool = False`

- [ ] **Step 1: 在 settings.py 中添加 InferenceConfig**

在 `LLMConfig` 之后、`GenerationConfig` 之前插入：

```python
class InferenceConfig(_BaseConfig):
    """本地 LLM 推理配置（llama-cpp-python + GGUF）"""
    llm_model: str = "Qwen/Qwen3-0.6B"
    gguf_file: str = "Qwen3-0.6B-Q4_K_M.gguf"
    n_ctx: int = Field(default=4096, ge=256)
    n_threads: int | None = None
    n_gpu_layers: int = Field(default=0, ge=0)
    default_max_tokens: int = Field(default=512, ge=1)
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    verbose: bool = False
```

- [ ] **Step 2: 更新 LLMConfig 添加 local_enabled**

在 `LLMConfig` 的 `local: str | None = None` 之后添加：

```python
class LLMConfig(_BaseConfig):
    # ... existing fields ...
    local: str | None = None
    local_enabled: bool = False    # 新增：启用本地 LLM
    # ... rest unchanged ...
```

- [ ] **Step 3: 在 RAGAppConfig 中注册 InferenceConfig**

在 `RAGAppConfig` 的字段列表中添加（在 `finetune` 之后）：

```python
class RAGAppConfig(BaseModel):
    # ... existing fields ...
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    # ... rest unchanged ...
```

- [ ] **Step 4: 更新 __all__ 导出**

在 `settings.py` 末尾的 `__all__` 列表中添加 `"InferenceConfig"`。

- [ ] **Step 5: 更新 config/__init__.py 导出**

```python
from config.settings import (
    # ... existing imports ...
    InferenceConfig,
)
```

- [ ] **Step 6: 更新 config/dev.yaml**

在 `llm:` 段之后、`generation:` 之前插入：

```yaml
# --------------------------------------------------------------------------
# 本地 LLM 推理（llama-cpp-python + GGUF）
# --------------------------------------------------------------------------
inference:
  llm_model: Qwen/Qwen3-0.6B
  gguf_file: Qwen3-0.6B-Q4_K_M.gguf
  n_ctx: 4096
  n_threads: null                    # null=自动检测 CPU 核心数
  n_gpu_layers: 0                    # 0=纯CPU运行
  default_max_tokens: 512
  default_temperature: 0.0           # 管线用（确定性）
  verbose: false
```

同时将 `llm.local:` 从 `null` 改为 `qwen3-0.6b`，并添加 `local_enabled: false`：

```yaml
llm:
  # ... existing ...
  local: qwen3-0.6b
  local_enabled: false
```

- [ ] **Step 7: 验证配置加载**

```bash
cd E:/Code/rag0709 && python -c "from config import settings; print(settings.inference.llm_model); print(settings.inference.n_ctx)"
```
Expected: `Qwen/Qwen3-0.6B` 和 `4096`

- [ ] **Step 8: Commit**

```bash
git add src/config/settings.py src/config/__init__.py config/dev.yaml
git commit -m "feat: add InferenceConfig for local LLM inference"
```

---

### Task 2: LocalLLM 类 — 核心推理引擎

**Files:**
- Modify: `src/model/inference.py` (add LocalLLM class)

**Interfaces:**
- Produces: `LocalLLM(model_path, n_ctx, n_threads, n_gpu_layers, verbose)` class
  - `model_path: Path` property
  - `is_loaded: bool` property
  - `load() -> None`
  - `__call__(prompt: str, **kwargs) -> str`
  - `stream(prompt: str, **kwargs) -> Iterator[str]`
  - `ainvoke(prompt: str, **kwargs) -> Awaitable[str]`
  - `unload() -> None`

- [ ] **Step 1: 在 inference.py 中添加 LocalLLM 类**

在文件末尾（`_reset_cache` 之前、测试辅助之前）插入完整的 `LocalLLM` 类：

```python
# ============================================================================
# 本地 LLM 推理引擎
# ============================================================================

_LOCAL_LLM_LOCK = threading.Lock()


class LocalLLM:
    """本地 LLM 推理引擎（llama-cpp-python + GGUF 量化模型）

    特性：
    - 懒加载：首次 __call__ 时才初始化 Llama 实例
    - 异步安全：ainvoke() 通过 asyncio.to_thread() 包装，不阻塞事件循环
    - 线程安全：推理调用受 threading.Lock 保护

    使用示例：
        llm = LocalLLM("models/Qwen3-0.6B-Q4_K_M.gguf")
        output = llm("你好，请介绍一下自己", max_tokens=512)
        # 流式
        for token in llm.stream("你好"):
            print(token, end="")
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._llm: Any = None  # llama_cpp.Llama 实例（懒加载）
        self._instance_lock = threading.Lock()

    # ---- 属性 ---------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # ---- 生命周期 -----------------------------------------------------------

    def load(self) -> None:
        """显式加载模型（通常无需调用，__call__ 会自动懒加载）"""
        if self._llm is not None:
            return
        with self._instance_lock:
            if self._llm is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python 未安装。请运行: pip install llama-cpp-python"
                )
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"GGUF 模型文件不存在: {self._model_path}\n"
                    f"请先下载模型: from model import models; "
                    f"models.download('{self._model_path.parent.name}')"
                )
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=self._verbose,
            )

    def unload(self) -> None:
        """释放模型资源"""
        with self._instance_lock:
            self._llm = None

    # ---- 推理 ---------------------------------------------------------------

    def __call__(self, prompt: str, **kwargs) -> str:
        """同步文本生成

        Args:
            prompt: 输入文本
            **kwargs: 透传给 Llama.create_completion()
                - max_tokens (int, default 512)
                - temperature (float, default 0.7)
                - top_p (float, default 0.95)
                - stop (list[str], default [])

        Returns:
            生成的文本字符串
        """
        self.load()
        max_tokens = kwargs.pop("max_tokens", 512)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.95)
        stop = kwargs.pop("stop", [])
        result = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )
        return result["choices"][0]["text"]

    def stream(self, prompt: str, **kwargs) -> "Iterator[str]":
        """流式文本生成

        Yields:
            增量生成的文本片段（token 级别）
        """
        self.load()
        max_tokens = kwargs.pop("max_tokens", 512)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.95)
        stop = kwargs.pop("stop", [])
        stream = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            content = chunk["choices"][0].get("text", "")
            if content:
                yield content

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """异步文本生成（通过 run_in_executor 包装同步调用）"""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self(prompt, **kwargs))
```

- [ ] **Step 2: 更新文件顶部 import**

在 inference.py 顶部添加 `from typing import Any, Iterator`：

```python
from typing import Any, Iterator
```

同时添加 `import asyncio` 到文件顶部（与现有 import 保持一致，但 asyncio 在 ainvoke 中延迟导入，不需要顶部添加）。

- [ ] **Step 3: 验证 LocalLLM 导入**

```bash
cd E:/Code/rag0709 && python -c "from model.inference import LocalLLM; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/model/inference.py
git commit -m "feat: add LocalLLM class for local GGUF inference"
```

---

### Task 3: get_local_llm() 单例工厂 + generate() 函数

**Files:**
- Modify: `src/model/inference.py` (add get_local_llm, update generate)

**Interfaces:**
- Produces: `get_local_llm() -> LocalLLM` — 进程级单例工厂
- Produces: `generate(prompt: str, **kwargs) -> str` — 委托给 get_local_llm()
- Consumes: `LocalLLM` (from Task 2), `InferenceConfig` (from Task 1)

- [ ] **Step 1: 在 LocalLLM 类之后添加 get_local_llm()**

```python
# ---- 进程级单例 ------------------------------------------------------------

_local_llm_instance: LocalLLM | None = None
_local_llm_lock = threading.Lock()


def get_local_llm() -> LocalLLM:
    """获取进程级 LocalLLM 单例（懒加载 + 双检锁）

    从 settings.inference 读取模型路径和参数。首次调用时自动加载配置。
    """
    global _local_llm_instance
    if _local_llm_instance is not None:
        return _local_llm_instance
    with _local_llm_lock:
        if _local_llm_instance is None:
            from config import settings
            from model import models

            cfg = settings.inference
            # 查找 GGUF 文件：优先本地路径，其次模型缓存目录
            gguf_path = _resolve_gguf_path(cfg, models)
            _local_llm_instance = LocalLLM(
                model_path=gguf_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads,
                n_gpu_layers=cfg.n_gpu_layers,
                verbose=cfg.verbose,
            )
    return _local_llm_instance


def _resolve_gguf_path(cfg: Any, models: Any) -> Path:
    """解析 GGUF 文件路径。

    查找顺序：
    1. 如果 gguf_file 是绝对路径 → 直接使用
    2. 在模型缓存目录下查找: local_models/{org}/{model_name}/{gguf_file}
    3. 抛出 FileNotFoundError（提示用户下载模型）
    """
    from pathlib import Path as _Path

    gguf_file = _Path(cfg.gguf_file)
    if gguf_file.is_absolute():
        return gguf_file

    # 尝试从模型缓存目录查找
    model_path = models.get_path(cfg.llm_model)
    if model_path is not None:
        candidate = model_path / cfg.gguf_file
        if candidate.exists():
            return candidate

    # 尝试 PROJECT_ROOT / local_models / org / model_name / gguf_file
    from config.path import PROJECT_ROOT
    parts = cfg.llm_model.split("/")
    fallback = PROJECT_ROOT / "local_models" / parts[0] / parts[1] / cfg.gguf_file
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"GGUF 模型文件未找到: {cfg.gguf_file}\n"
        f"请先下载 {cfg.llm_model} 模型，并将 GGUF 文件放入模型目录，"
        f"或设置绝对路径: settings.apply_overrides('inference.gguf_file=/path/to/model.gguf')"
    )
```

- [ ] **Step 2: 替换 generate() 函数**

将当前的 `generate()` 替换为：

```python
def generate(prompt: str, **kwargs) -> str:
    """LLM 文本生成（本地 llama-cpp-python + GGUF 推理）

    首次调用时自动加载模型（懒加载），后续调用复用已加载的实例。

    Args:
        prompt: 输入提示文本
        **kwargs: 透传给 LocalLLM.__call__()
            - max_tokens: 最大生成 token 数（默认 512）
            - temperature: 温度（默认从 settings.inference.default_temperature）
            - stop: 停止词列表

    Returns:
        生成的文本

    Raises:
        ImportError: llama-cpp-python 未安装
        FileNotFoundError: GGUF 模型文件不存在
    """
    llm = get_local_llm()
    # 使用配置中的默认 temperature，除非调用方显式指定
    if "temperature" not in kwargs:
        from config import settings
        kwargs["temperature"] = settings.inference.default_temperature
    if "max_tokens" not in kwargs:
        from config import settings
        kwargs["max_tokens"] = settings.inference.default_max_tokens
    return llm(prompt, **kwargs)
```

- [ ] **Step 3: 更新 _reset_cache() 以支持测试隔离**

```python
def _reset_cache() -> None:
    """重置模块级模型缓存（仅用于测试隔离）"""
    global _embedding_model, _cross_encoder, _local_llm_instance
    _embedding_model = None
    _cross_encoder = None
    _local_llm_instance = None
```

- [ ] **Step 4: 验证**

```bash
cd E:/Code/rag0709 && python -c "from model.inference import get_local_llm, generate; print('OK')"
```
Expected: `OK`（无需实际加载模型）

- [ ] **Step 5: Commit**

```bash
git add src/model/inference.py
git commit -m "feat: add get_local_llm() singleton + generate() for local LLM"
```

---

### Task 4: LocalLLMAdapter — LLMProtocol 适配器

**Files:**
- Create: `src/model/llm_adapter.py`

**Interfaces:**
- Produces: `LocalLLMAdapter` class implementing `LLMProtocol`
  - `__init__(self, llm: LocalLLM, default_temperature: float = 0.0)`
  - `async ainvoke(self, prompt: str, **kwargs) -> _FakeMessage`
- Produces: `_FakeMessage` — 简单的 `.content` 容器

- [ ] **Step 1: 创建 llm_adapter.py**

```python
"""LocalLLMAdapter — 将 LocalLLM 适配为 LLMProtocol，无缝接入现有生成管线

使用示例：
    from model.inference import LocalLLM
    from model.llm_adapter import LocalLLMAdapter

    llm = LocalLLM("models/Qwen3-0.6B-Q4_K_M.gguf")
    adapter = LocalLLMAdapter(llm)

    # 直接替换云端 LLM，注入 Pipeline
    from core import get_rag_pipeline
    pipeline = get_rag_pipeline(adapter, session_manager)
    ctx = await pipeline.run("什么是RAG？")
"""

from typing import Any


class _FakeMessage:
    """模拟 LangChain BaseMessage，提供 .content 属性"""

    def __init__(self, content: str):
        self.content = content

    def __repr__(self) -> str:
        return f"_FakeMessage(content={self.content[:50]!r}...)"


class LocalLLMAdapter:
    """将 LocalLLM 适配为 LLMProtocol

    符合 models.llm.LLMProtocol 协议：
    - async ainvoke(prompt, **kwargs) -> object with .content

    可直接注入 GenerationLayer / RAGPipeline / FactChecker 等需要 LLMProtocol 的组件。
    """

    def __init__(self, llm: "LocalLLM", default_temperature: float = 0.0):
        """
        Args:
            llm: LocalLLM 实例（可以是已加载或未加载的）
            default_temperature: 默认温度（优先级低于 kwargs 显式传入的 temperature）
        """
        self._llm = llm
        self._default_temperature = default_temperature

    @property
    def llm(self) -> "LocalLLM":
        return self._llm

    async def ainvoke(self, prompt: str, **kwargs: Any) -> _FakeMessage:
        """异步 LLM 调用，返回 _FakeMessage（有 .content 属性）

        Args:
            prompt: 输入提示
            **kwargs: 透传给 LocalLLM.ainvoke()
                - temperature: 温度（默认 self._default_temperature）
                - max_tokens: 最大 token 数

        Returns:
            _FakeMessage(content=generated_text)
        """
        temperature = kwargs.pop("temperature", self._default_temperature)
        content = await self._llm.ainvoke(prompt, temperature=temperature, **kwargs)
        return _FakeMessage(content)
```

- [ ] **Step 2: 验证导入**

```bash
cd E:/Code/rag0709 && python -c "from model.llm_adapter import LocalLLMAdapter, _FakeMessage; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/model/llm_adapter.py
git commit -m "feat: add LocalLLMAdapter implementing LLMProtocol"
```

---

### Task 5: download_gguf() — GGUF 文件下载

**Files:**
- Modify: `src/model/downloader.py` (add download_gguf method)

**Interfaces:**
- Produces: `ModelDownloader.download_gguf(model_id: str, gguf_filename: str) -> Path`

- [ ] **Step 1: 在 ModelDownloader 类中添加 download_gguf()**

```python
def download_gguf(self, model_id: str, gguf_filename: str) -> Path:
    """下载指定的 GGUF 文件到模型目录。

    利用 huggingface_hub.hf_hub_download 下载单个文件，
    自动继承 cache_dir / endpoint / token 配置，支持断点续传和重试。

    Args:
        model_id: HuggingFace repo_id，如 "Qwen/Qwen3-0.6B"
        gguf_filename: GGUF 文件名，如 "Qwen3-0.6B-Q4_K_M.gguf"

    Returns:
        下载后的本地文件路径

    Raises:
        RuntimeError: 下载失败（超过 max_retries）
    """
    model_id = _validate_model_id(model_id)
    # 确保模型目录存在
    model_dir = self._cache_dir / model_id.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    delay = 1.0
    for attempt in range(self._max_retries + 1):
        try:
            local_path = hf_hub_download(
                repo_id=model_id,
                filename=gguf_filename,
                local_dir=model_dir,
                token=self._hf_token,
                endpoint=self._huggingface_endpoint,
                resume_download=True,
            )
            logger.info(
                f"GGUF 文件已下载: {gguf_filename} → {local_path}"
            )
            return Path(local_path)
        except (HfHubHTTPError, RepositoryNotFoundError) as e:
            last_error = e
            if attempt == self._max_retries:
                break
            logger.warning(
                f"GGUF 下载失败 (attempt {attempt + 1}/{self._max_retries + 1}): "
                f"{e}，{delay:.0f}s 后重试"
            )
            time.sleep(delay)
            delay *= 2
        except OSError as e:
            last_error = e
            if attempt == self._max_retries:
                break
            logger.warning(
                f"GGUF 下载 IO 错误 (attempt {attempt + 1}/{self._max_retries + 1}): "
                f"{e}，{delay:.0f}s 后重试"
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(
        f"GGUF 文件下载失败（已重试 {self._max_retries} 次）: {last_error}"
    )
```

需要在文件顶部添加 `hf_hub_download` 导入：

```python
from huggingface_hub import hf_hub_download, snapshot_download
```

- [ ] **Step 2: 验证导入**

```bash
cd E:/Code/rag0709 && python -c "from model.downloader import ModelDownloader; print(hasattr(ModelDownloader, 'download_gguf'))"
```
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add src/model/downloader.py
git commit -m "feat: add download_gguf() for single GGUF file download"
```

---

### Task 6: Manager & __init__.py 集成

**Files:**
- Modify: `src/model/manager.py` (add local_llm property)
- Modify: `src/model/__init__.py` (export new components)

**Interfaces:**
- Produces: `models.local_llm` property → LocalLLM instance
- Produces: Updated `models.generate()` → delegates to inference.generate()

- [ ] **Step 1: 在 ModelManager 中添加 local_llm 属性**

在 `manager.py` 的 `ModelManager` 类中，`cross_encoder` 属性之后添加：

```python
@property
def local_llm(self):
    """获取 LocalLLM 推理引擎实例（懒加载 + 双检锁）

    首次访问时从 settings.inference 读取配置并创建实例。
    """
    from . import inference as _inference
    return _inference.get_local_llm()
```

- [ ] **Step 2: models.generate() 已存在，无需修改**

当前 `models.generate()` 已经委托给 `inference.generate()`，Task 3 中 `generate()` 已实现真实逻辑，此处无需改动。

- [ ] **Step 3: 更新 __init__.py 导出**

在 `src/model/__init__.py` 中，更新 import 和 __all__：

```python
"""
模型管理模块 — 统一下载和管理 embedding / rerank / 本地 LLM 模型，
以及微调和蒸馏训练。

使用示例：
    from model import models

    # 本地 LLM 推理
    output = models.generate("你好，请介绍一下自己")
    llm = models.local_llm          # 获取 LocalLLM 实例

    # 适配器：接入管线
    from model.llm_adapter import LocalLLMAdapter
    adapter = LocalLLMAdapter(models.local_llm)
"""
__version__ = "1.2.0"

from . import inference
from .downloader import (
    AutoStrategy,
    DownloadStrategy,
    HfStrategy,
    ModelDownloader,
    MsStrategy,
)
from .inference import LocalLLM, generate, get_local_llm
from .llm_adapter import LocalLLMAdapter
from .manager import ModelManager, models

__all__ = [
    "AutoStrategy",
    "DownloadStrategy",
    "HfStrategy",
    "LocalLLM",
    "LocalLLMAdapter",
    "ModelDownloader",
    "ModelManager",
    "MsStrategy",
    "__version__",
    "generate",
    "get_local_llm",
    "inference",
    "models",
]
```

- [ ] **Step 4: 验证导入**

```bash
cd E:/Code/rag0709 && python -c "from model import models, LocalLLM, LocalLLMAdapter, get_local_llm; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/model/manager.py src/model/__init__.py
git commit -m "feat: integrate local LLM into ModelManager and exports"
```

---

### Task 7: proxy/models.py — Pydantic 请求/响应模型

**Files:**
- Create: `src/model/proxy/__init__.py`
- Create: `src/model/proxy/models.py`

**Interfaces:**
- Produces: `ChatCompletionRequest`, `Message`, `ChatCompletionResponse`, `Choice`, `Usage`, `ModelList`, `ModelInfo`

- [ ] **Step 1: 创建 proxy/__init__.py**

```python
"""本地 LLM OpenAI 兼容代理服务"""
```

- [ ] **Step 2: 创建 proxy/models.py**

```python
"""OpenAI Chat Completions API 兼容的 Pydantic 模型"""

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions 请求模型（子集）"""

    model: str = "local-llm"
    messages: list[Message]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False

    def to_kwargs(self) -> dict:
        """提取非空推理参数"""
        kwargs: dict = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stop is not None:
            kwargs["stop"] = self.stop
        return kwargs


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = "local-llm-completion"
    object: str = "chat.completion"
    created: int = 0
    model: str = "local-llm"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
```

- [ ] **Step 3: 验证**

```bash
cd E:/Code/rag0709 && python -c "from model.proxy.models import ChatCompletionRequest, ChatCompletionResponse; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/model/proxy/
git commit -m "feat: add proxy Pydantic models for OpenAI API compat"
```

---

### Task 8: proxy/server.py + __main__.py — FastAPI 服务

**Files:**
- Create: `src/model/proxy/server.py`
- Create: `src/model/proxy/__main__.py`

**Interfaces:**
- Produces: `app` — FastAPI application
  - `GET /v1/models` → `ModelList`
  - `POST /v1/chat/completions` → `ChatCompletionResponse`
  - `GET /health` → `{"status": "ok"}`
- Produces: `__main__.py` — CLI entry `python -m model.proxy`

- [ ] **Step 1: 创建 proxy/server.py**

```python
"""本地 LLM OpenAI 兼容代理服务 — FastAPI 子应用

启动: python -m model.proxy --port 8080
LangChain 集成: ChatOpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
"""

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from model.inference import get_local_llm
from model.proxy.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    ModelInfo,
    ModelList,
    Usage,
)

app = FastAPI(
    title="Local LLM Proxy",
    description="OpenAI-compatible API for local llama-cpp-python LLM",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelInfo(id="local-llm")])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    llm = get_local_llm()
    # 将 messages 列表拼接为单个 prompt（简化实现；后续可扩展 chat template）
    prompt = _messages_to_prompt(request.messages)
    kwargs = request.to_kwargs()
    try:
        content = await llm.ainvoke(prompt, **kwargs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


def _messages_to_prompt(messages: list[Message]) -> str:
    """将 chat messages 列表拼接为单轮 prompt

    简化实现（非 chat template）：
    - 单条消息 → 直接返回 content
    - 多条消息 → 用角色标签拼接
    """
    if len(messages) == 1:
        return messages[0].content
    parts: list[str] = []
    for msg in messages:
        role = msg.role
        if role == "system":
            parts.append(f"<|system|>\n{msg.content}\n</|system|>")
        elif role == "user":
            parts.append(f"<|user|>\n{msg.content}\n</|user|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}\n</|assistant|>")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )
```

- [ ] **Step 2: 创建 proxy/__main__.py**

```python
"""本地 LLM OpenAI 兼容代理服务 CLI 入口

用法:
    python -m model.proxy --port 8080
    python -m model.proxy --port 8080 --model /path/to/model.gguf
    python -m model.proxy --host 0.0.0.0 --port 8080
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="本地 LLM OpenAI 兼容代理服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m model.proxy --port 8080
  python -m model.proxy --model local_models/Qwen3-0.6B-Q4_K_M.gguf
  python -m model.proxy --host 0.0.0.0 --port 8080

LangChain 集成:
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
""",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")
    parser.add_argument("--model", default=None, help="GGUF 模型路径（覆盖配置文件）")
    parser.add_argument("--n-ctx", type=int, default=None, help="上下文窗口大小")
    args = parser.parse_args()

    # 添加 src 目录到 sys.path（确保 config 等模块可导入）
    src_path = Path(__file__).resolve().parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # 如果指定了 model 参数，覆盖配置
    if args.model:
        from config import settings
        settings.apply_overrides(f"inference.gguf_file={args.model}")
    if args.n_ctx:
        from config import settings
        settings.apply_overrides(f"inference.n_ctx={args.n_ctx}")

    import uvicorn
    from model.proxy.server import app

    print(f"Local LLM Proxy starting on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"Endpoint:  http://{args.host}:{args.port}/v1/chat/completions")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证代理服务可以导入**

```bash
cd E:/Code/rag0709 && python -c "from model.proxy.server import app; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/model/proxy/server.py src/model/proxy/__main__.py
git commit -m "feat: add OpenAI-compatible proxy server + CLI entry"
```

---

### Task 9: pyproject.toml 依赖更新

**Files:**
- Modify: `pyproject.toml` (add local-llm optional dependency)

- [ ] **Step 1: 在 pyproject.toml 中添加 local-llm 依赖组**

在 `[project.optional-dependencies]` 中，`finetune` 之后添加：

```toml
# 本地 LLM 推理（llama-cpp-python + GGUF）
local-llm = [
    "llama-cpp-python>=0.3.0",
]
```

并更新 `all` 依赖组，将 `local-llm` 加入：

```toml
all = [
    "rag-service[retrieval,ingestion,dev,finetune,fallback,local-llm]",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add local-llm optional dependency (llama-cpp-python)"
```

---

### Task 10: 测试

**Files:**
- Modify: `tests/unit/model/test_inference.py` (update generate tests, add LocalLLM tests)
- Create: `tests/unit/model/test_llm_adapter.py`
- Create: `tests/unit/model/test_proxy.py`

- [ ] **Step 1: 更新 test_inference.py 中的 TestGenerate 类**

替换现有的 `TestGenerate` 类：

```python
class _FakeLlama:
    """模拟 llama-cpp-python 的 Llama 实例"""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def create_completion(self, prompt, max_tokens=512, temperature=0.7,
                          top_p=0.95, stop=None, stream=False, **kwargs):
        if stream:
            return iter([{"choices": [{"text": "你好"}]}, {"choices": [{"text": "！"}]}])
        return {"choices": [{"text": "这是模拟的LLM回复。"}]}


class TestGenerate:
    def test_generate_returns_string(self, monkeypatch):
        """generate() 返回字符串"""
        monkeypatch.setattr(
            inference, "get_local_llm",
            lambda: inference.LocalLLM("/fake/model.gguf")
        )
        monkeypatch.setattr(
            "llama_cpp.Llama", _FakeLlama, raising=False
        )
        # 直接操作 _local_llm_instance 来跳过文件检查
        import model.inference as _inf
        with _inf._local_llm_lock:
            _inf._local_llm_instance = _inf.LocalLLM.__new__(_inf.LocalLLM)
            _inf._local_llm_instance._model_path = Path("/fake/model.gguf")
            _inf._local_llm_instance._n_ctx = 4096
            _inf._local_llm_instance._n_threads = None
            _inf._local_llm_instance._n_gpu_layers = 0
            _inf._local_llm_instance._verbose = False
            _inf._local_llm_instance._instance_lock = threading.Lock()
            _inf._local_llm_instance._llm = _FakeLlama()
        try:
            result = inference.generate("测试prompt", temperature=0.5, max_tokens=100)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            _inf._local_llm_instance = None

    def test_generate_kwargs_passthrough(self, monkeypatch):
        """**kwargs 透传给底层模型"""
        captured: list = []

        class _CaptureLlama:
            def create_completion(self, **kwargs):
                captured.append(kwargs)
                return {"choices": [{"text": "ok"}]}

        monkeypatch.setattr("llama_cpp.Llama", _CaptureLlama, raising=False)
        import model.inference as _inf
        with _inf._local_llm_lock:
            _inf._local_llm_instance = _inf.LocalLLM.__new__(_inf.LocalLLM)
            _inf._local_llm_instance._model_path = Path("/fake/model.gguf")
            _inf._local_llm_instance._n_ctx = 4096
            _inf._local_llm_instance._n_threads = None
            _inf._local_llm_instance._n_gpu_layers = 0
            _inf._local_llm_instance._verbose = False
            _inf._local_llm_instance._instance_lock = threading.Lock()
            _inf._local_llm_instance._llm = _CaptureLlama()
        try:
            result = inference.generate("测试", temperature=0.3, max_tokens=200, top_p=0.8)
            assert result == "ok"
            assert captured[0]["temperature"] == 0.3
            assert captured[0]["max_tokens"] == 200
        finally:
            _inf._local_llm_instance = None

    def test_local_llm_lazy_load(self):
        """LocalLLM 构造时不加载，__call__ 时才加载"""
        llm = inference.LocalLLM("/nonexistent/model.gguf")
        assert not llm.is_loaded
        assert llm.model_path == Path("/nonexistent/model.gguf")

    def test_local_llm_file_not_found(self):
        """GGUF 文件不存在时 __call__ 抛 FileNotFoundError"""
        llm = inference.LocalLLM("/nonexistent/model.gguf")
        with pytest.raises(FileNotFoundError):
            llm("测试")

    def test_generate_uses_defaults_from_config(self, monkeypatch):
        """generate() 使用 settings.inference 中的默认参数"""
        monkeypatch.setattr("llama_cpp.Llama", _FakeLlama, raising=False)
        import model.inference as _inf
        with _inf._local_llm_lock:
            _inf._local_llm_instance = _inf.LocalLLM.__new__(_inf.LocalLLM)
            _inf._local_llm_instance._model_path = Path("/fake/model.gguf")
            _inf._local_llm_instance._n_ctx = 4096
            _inf._local_llm_instance._n_threads = None
            _inf._local_llm_instance._n_gpu_layers = 0
            _inf._local_llm_instance._verbose = False
            _inf._local_llm_instance._instance_lock = threading.Lock()
            _inf._local_llm_instance._llm = _FakeLlama()
        try:
            result = inference.generate("测试")
            assert isinstance(result, str)
        finally:
            _inf._local_llm_instance = None

    def test_get_local_llm_singleton(self, monkeypatch):
        """get_local_llm() 返回同一实例"""
        monkeypatch.setattr(
            inference, "_resolve_gguf_path",
            lambda cfg, models: Path("/fake/model.gguf")
        )
        monkeypatch.setattr("llama_cpp.Llama", _FakeLlama, raising=False)
        inference._reset_cache()
        try:
            llm1 = inference.get_local_llm()
            llm2 = inference.get_local_llm()
            assert llm1 is llm2
        finally:
            inference._reset_cache()

    def test_local_llm_stream(self, monkeypatch):
        """stream() 方法返回迭代器"""
        llm = inference.LocalLLM.__new__(inference.LocalLLM)
        llm._model_path = Path("/fake/model.gguf")
        llm._n_ctx = 4096
        llm._n_threads = None
        llm._n_gpu_layers = 0
        llm._verbose = False
        llm._instance_lock = threading.Lock()
        llm._llm = _FakeLlama()
        tokens = list(llm.stream("你好"))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_local_llm_unload(self, monkeypatch):
        """unload() 后 is_loaded 为 False"""
        llm = inference.LocalLLM.__new__(inference.LocalLLM)
        llm._model_path = Path("/fake/model.gguf")
        llm._n_ctx = 4096
        llm._instance_lock = threading.Lock()
        llm._llm = _FakeLlama()
        assert llm.is_loaded
        llm.unload()
        assert not llm.is_loaded
```

- [ ] **Step 2: 创建 test_llm_adapter.py**

```python
"""LocalLLMAdapter 测试"""
import pytest

from model.llm_adapter import LocalLLMAdapter, _FakeMessage


class _FakeLocalLLM:
    """模拟 LocalLLM，可直接记录调用参数"""

    def __init__(self):
        self.calls: list[dict] = []

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return f"回复: {prompt[:20]}"


class TestLocalLLMAdapter:
    def test_ainvoke_returns_fake_message(self):
        """ainvoke() 返回 _FakeMessage，有 .content 属性"""
        fake_llm = _FakeLocalLLM()
        adapter = LocalLLMAdapter(fake_llm)
        result = adapter.ainvoke("你好")
        # 需要 asyncio 运行
        import asyncio
        msg = asyncio.run(result)
        assert isinstance(msg, _FakeMessage)
        assert msg.content == "回复: 你好"

    def test_ainvoke_passes_temperature(self):
        """ainvoke() 透传 temperature"""
        fake_llm = _FakeLocalLLM()
        adapter = LocalLLMAdapter(fake_llm)
        import asyncio
        asyncio.run(adapter.ainvoke("测试", temperature=0.5))
        assert fake_llm.calls[0]["temperature"] == 0.5

    def test_default_temperature(self):
        """未指定 temperature 时使用默认值"""
        fake_llm = _FakeLocalLLM()
        adapter = LocalLLMAdapter(fake_llm, default_temperature=0.3)
        import asyncio
        asyncio.run(adapter.ainvoke("测试"))
        assert fake_llm.calls[0]["temperature"] == 0.3

    def test_llm_property(self):
        """llm 属性返回注入的 LocalLLM"""
        fake_llm = _FakeLocalLLM()
        adapter = LocalLLMAdapter(fake_llm)
        assert adapter.llm is fake_llm

    def test_fake_message_repr(self):
        """_FakeMessage.__repr__ 包含内容预览"""
        msg = _FakeMessage("这是一条很长的消息内容" * 10)
        assert "..." in repr(msg)
```

- [ ] **Step 3: 创建 test_proxy.py**

```python
"""代理服务测试"""
import pytest
from fastapi.testclient import TestClient

from model.proxy.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestProxyHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestProxyModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["data"][0]["id"] == "local-llm"


class TestProxyChatCompletions:
    def test_chat_completions_structure(self, client, monkeypatch):
        """验证 chat completions 响应结构（mock LLM）"""
        class _FakeLLM:
            async def ainvoke(self, prompt, **kwargs):
                return "测试回复"

        monkeypatch.setattr(
            "model.proxy.server.get_local_llm",
            lambda: _FakeLLM()
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "local-llm",
            "messages": [{"role": "user", "content": "你好"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "测试回复"
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completions_with_params(self, client, monkeypatch):
        """验证参数透传"""
        captured = {}

        class _FakeLLM:
            async def ainvoke(self, prompt, **kwargs):
                captured.update(kwargs)
                return "ok"

        monkeypatch.setattr(
            "model.proxy.server.get_local_llm",
            lambda: _FakeLLM()
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "local-llm",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0.5,
            "max_tokens": 256,
        })
        assert resp.status_code == 200
        assert captured.get("temperature") == 0.5
        assert captured.get("max_tokens") == 256

    def test_chat_completions_file_not_found(self, client, monkeypatch):
        """LLM 加载失败返回 503"""
        class _FakeLLM:
            async def ainvoke(self, prompt, **kwargs):
                raise FileNotFoundError("model.gguf not found")

        monkeypatch.setattr(
            "model.proxy.server.get_local_llm",
            lambda: _FakeLLM()
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "local-llm",
            "messages": [{"role": "user", "content": "测试"}],
        })
        assert resp.status_code == 503
```

- [ ] **Step 4: 运行测试**

```bash
cd E:/Code/rag0709 && python -m pytest tests/unit/model/test_inference.py tests/unit/model/test_llm_adapter.py tests/unit/model/test_proxy.py -v
```
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/model/test_inference.py tests/unit/model/test_llm_adapter.py tests/unit/model/test_proxy.py
git commit -m "test: add tests for LocalLLM, adapter, and proxy"
```
