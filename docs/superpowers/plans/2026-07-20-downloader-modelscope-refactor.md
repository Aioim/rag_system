# Model Downloader 重构 — 集成魔搭下载 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ModelScope 下载集成到 `ModelDownloader` 类中，通过策略模式统一 HF/hf-mirror 和魔搭下载路径，支持全局配置选择下载源。

**Architecture:** 策略模式 — `ModelDownloader` 持有 `DownloadStrategy` 协议实例，`download()` 委托给策略。`HfStrategy`（现有 HF 逻辑）、`MsStrategy`（现有魔搭逻辑）、`AutoStrategy`（组合两者，先 MS 后 HF fallback）。`is_downloaded` / `list_downloaded` / `remove` / `model_dir` 保持为通用本地文件操作。

**Tech Stack:** Python 3.11+, huggingface_hub, modelscope (可选依赖), pytest + pytest-asyncio

## Global Constraints

- 公共接口不变：`ModelDownloader.download()` 签名和返回值不变，`ModelManager` 对外 API 完全不变
- `modelscope` 为可选依赖，未安装时 `MsStrategy.__init__` 抛 `RuntimeError`
- `download_source` 不合法值时 `ModelDownloader._build_strategy` 抛 `ValueError`
- 删除独立函数 `download_from_modelscope()` 和模块级 `_MODELSCOPE_AVAILABLE`
- 遵循 PEP 8 + 类型注解 + black/isort/ruff 格式

---

### Task 1: 配置 — 新增 `download_source` 字段

**Files:**
- Modify: `src/config/settings.py:221-231`
- Modify: `config/dev.yaml:144-152`

**Interfaces:**
- Consumes: 无
- Produces: `ModelConfig.download_source: str = "auto"`

- [ ] **Step 1: 在 `settings.py` 的 `ModelConfig` 中新增字段**

```python
# src/config/settings.py — ModelConfig 类，在 max_retries 之后新增一行
class ModelConfig(_BaseConfig):
    """模型下载管理配置"""
    cache_dir: Path = PROJECT_ROOT / "local_models"
    default_models: dict[str, str] = Field(default_factory=lambda: {
        "embedding": "BAAI/bge-large-zh-v1.5",
        "rerank": "BAAI/bge-reranker-v2-m3",
        "llm": "Qwen/Qwen2.5-1.5B-Instruct",
    })
    hf_token_env: str = "HUGGINGFACE_TOKEN"
    hf_endpoint: str | None = "https://hf-mirror.com"
    max_retries: int = 3
    download_source: str = "auto"  # "huggingface" | "modelscope" | "auto"
```

- [ ] **Step 2: 在 `config/dev.yaml` 的 `model:` 段中新增配置**

```yaml
# config/dev.yaml — model 段，在 max_retries 之后新增一行
model:
  cache_dir: local_models
  default_models:
    embedding: BAAI/bge-large-zh-v1.5
    rerank: BAAI/bge-reranker-v2-m3
    llm: Qwen/Qwen2.5-1.5B-Instruct
  hf_token_env: HUGGINGFACE_TOKEN
  hf_endpoint: https://hf-mirror.com
  max_retries: 3
  download_source: auto                   # huggingface | modelscope | auto
```

- [ ] **Step 3: 验证配置可正常加载**

```bash
cd "E:\Code\rag0709" && python -c "from config import settings; print('download_source:', settings.model.download_source)"
```

Expected: `download_source: auto`

- [ ] **Step 4: Commit**

```bash
git add src/config/settings.py config/dev.yaml
git commit -m "feat(config): add model.download_source field for download source selection"
```

---

### Task 2: 测试 — 编写策略与 Downloader 测试

**Files:**
- Create: `tests/unit/model/test_downloader.py`

**Interfaces:**
- Consumes: `ModelConfig.download_source: str`（Task 1），现有 `ModelDownloader`、`_validate_model_id`
- Produces: 测试覆盖 `HfStrategy` / `MsStrategy` / `AutoStrategy` / `ModelDownloader.download()`

- [ ] **Step 1: 创建测试文件骨架**

```python
# tests/unit/model/test_downloader.py
"""测试 downloader 模块：策略类 + ModelDownloader"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.model.downloader import (
    _validate_model_id,
    HfStrategy,
    MsStrategy,
    AutoStrategy,
    ModelDownloader,
)


class TestValidateModelId:
    """模块级校验函数 — 行为不变"""

    def test_valid_id(self):
        assert _validate_model_id("BAAI/bge-large-zh-v1.5") == "BAAI/bge-large-zh-v1.5"

    def test_rejects_backslash(self):
        with pytest.raises(ValueError, match="非法字符"):
            _validate_model_id("BAAI\\bge-large-zh-v1.5")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="路径穿越"):
            _validate_model_id("../etc/passwd")

    def test_rejects_absolute_path(self):
        with pytest.raises(ValueError, match="绝对路径"):
            _validate_model_id("/etc/models")


class TestHfStrategy:
    """HuggingFace 下载策略"""

    def test_download_calls_snapshot_download(self, tmp_path):
        """验证 snapshot_download 被正确调用"""
        with patch("src.model.downloader.snapshot_download") as mock_sd:
            strategy = HfStrategy(endpoint=None, token=None, max_retries=1)
            result = strategy.download("org/model", force=False, cache_dir=tmp_path)

            mock_sd.assert_called_once()
            call_kwargs = mock_sd.call_args.kwargs
            assert call_kwargs["repo_id"] == "org/model"
            assert call_kwargs["local_dir"] == str(tmp_path / "org" / "model")

    def test_download_passes_endpoint_and_token(self, tmp_path):
        """验证 endpoint 和 token 传递"""
        with patch("src.model.downloader.snapshot_download") as mock_sd:
            strategy = HfStrategy(
                endpoint="https://hf-mirror.com",
                token="hf_test123",
                max_retries=1,
            )
            strategy.download("org/model", force=False, cache_dir=tmp_path)

            call_kwargs = mock_sd.call_args.kwargs
            assert call_kwargs["token"] == "hf_test123"
            assert call_kwargs["endpoint"] == "https://hf-mirror.com"

    def test_retries_on_os_error(self, tmp_path):
        """验证网络错误时重试"""
        with patch("src.model.downloader.snapshot_download") as mock_sd:
            mock_sd.side_effect = [OSError("network"), MagicMock()]
            strategy = HfStrategy(max_retries=3)
            result = strategy.download("org/model", force=False, cache_dir=tmp_path)
            assert mock_sd.call_count == 2

    def test_raises_after_max_retries(self, tmp_path):
        """验证超过重试次数后抛出 RuntimeError"""
        with patch("src.model.downloader.snapshot_download") as mock_sd:
            mock_sd.side_effect = OSError("persistent network error")
            strategy = HfStrategy(max_retries=2)
            with pytest.raises(RuntimeError, match="模型下载失败"):
                strategy.download("org/model", force=False, cache_dir=tmp_path)

    def test_raises_value_error_for_missing_repo(self, tmp_path):
        """验证仓库不存在时抛出 ValueError"""
        from huggingface_hub.utils import RepositoryNotFoundError

        with patch("src.model.downloader.snapshot_download") as mock_sd:
            mock_sd.side_effect = RepositoryNotFoundError("not found", response=None)
            strategy = HfStrategy(max_retries=1)
            with pytest.raises(ValueError, match="模型仓库不存在"):
                strategy.download("org/nonexistent", force=False, cache_dir=tmp_path)


class TestMsStrategy:
    """ModelScope 下载策略"""

    def test_raises_if_modelscope_not_installed(self):
        """验证 modelscope 未安装时抛 RuntimeError"""
        with patch.dict("sys.modules", {"modelscope": None}):
            with pytest.raises(RuntimeError, match="modelscope 未安装"):
                MsStrategy()

    def test_download_calls_modelscope_snapshot(self, tmp_path):
        """验证调用 modelscope.snapshot_download"""
        mock_ms = MagicMock()
        mock_ms_sd = MagicMock(return_value=str(tmp_path / ".modelscope" / "org--model"))
        mock_ms.snapshot_download = mock_ms_sd

        with patch.dict("sys.modules", {"modelscope": mock_ms}):
            strategy = MsStrategy()
            # 手动替换 _snapshot_download 引用
            strategy._snapshot_download = mock_ms_sd

            result = strategy.download("org/model", force=False, cache_dir=tmp_path)
            mock_ms_sd.assert_called_once_with(
                "org/model",
                cache_dir=str(tmp_path / ".modelscope_cache"),
            )

    def test_download_cleans_up_on_failure(self, tmp_path):
        """验证下载失败时清理不完整目录"""
        mock_ms_sd = MagicMock(side_effect=RuntimeError("download failed"))

        with patch.dict("sys.modules", {"modelscope": MagicMock()}):
            strategy = MsStrategy()
            strategy._snapshot_download = mock_ms_sd

            with pytest.raises(RuntimeError, match="ModelScope 下载失败"):
                strategy.download("org/model", force=False, cache_dir=tmp_path)


class TestAutoStrategy:
    """自动选择策略 — 优先 MS，失败回退 HF"""

    def test_uses_ms_when_successful(self, tmp_path):
        """MS 成功时不调用 HF"""
        ms = MagicMock()
        ms.download.return_value = tmp_path / "org" / "model"
        hf = MagicMock()

        strategy = AutoStrategy(ms, hf)
        result = strategy.download("org/model", force=False, cache_dir=tmp_path)

        ms.download.assert_called_once()
        hf.download.assert_not_called()

    def test_falls_back_to_hf_when_ms_fails(self, tmp_path):
        """MS 失败时回退到 HF"""
        ms = MagicMock()
        ms.download.side_effect = RuntimeError("MS failed")
        hf = MagicMock()
        hf.download.return_value = tmp_path / "org" / "model"

        strategy = AutoStrategy(ms, hf)
        result = strategy.download("org/model", force=False, cache_dir=tmp_path)

        ms.download.assert_called_once()
        hf.download.assert_called_once_with("org/model", False, tmp_path)

    def test_falls_back_when_ms_not_available(self, tmp_path):
        """modelscope 不可用时回退 HF"""
        ms = MagicMock()
        ms.download.side_effect = ImportError("no modelscope")
        hf = MagicMock()
        hf.download.return_value = tmp_path / "org" / "model"

        strategy = AutoStrategy(ms, hf)
        result = strategy.download("org/model", force=False, cache_dir=tmp_path)

        hf.download.assert_called_once()


class TestModelDownloader:
    """ModelDownloader 集成测试"""

    def test_builds_hf_strategy(self, tmp_path):
        """download_source='huggingface' 创建 HfStrategy"""
        dl = ModelDownloader(
            cache_dir=tmp_path,
            download_source="huggingface",
            hf_token="t",
        )
        assert isinstance(dl._strategy, HfStrategy)

    def test_builds_ms_strategy(self, tmp_path):
        """download_source='modelscope' 创建 MsStrategy"""
        dl = ModelDownloader(
            cache_dir=tmp_path,
            download_source="modelscope",
        )
        assert isinstance(dl._strategy, MsStrategy)

    def test_builds_auto_strategy(self, tmp_path):
        """download_source='auto' 创建 AutoStrategy（默认）"""
        dl = ModelDownloader(cache_dir=tmp_path)
        assert isinstance(dl._strategy, AutoStrategy)

    def test_build_strategy_rejects_invalid_source(self, tmp_path):
        """非法 download_source 抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的 download_source"):
            ModelDownloader(cache_dir=tmp_path, download_source="invalid")

    def test_download_skips_when_exists(self, tmp_path):
        """已下载模型跳过下载"""
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("weight")

        mock_strategy = MagicMock()
        dl = ModelDownloader(cache_dir=tmp_path)
        dl._strategy = mock_strategy

        result = dl.download("org/model")
        mock_strategy.download.assert_not_called()
        assert result == model_dir

    def test_download_delegates_to_strategy(self, tmp_path):
        """未下载时委托给策略"""
        mock_strategy = MagicMock()
        mock_strategy.download.return_value = tmp_path / "org" / "model"

        dl = ModelDownloader(cache_dir=tmp_path)
        dl._strategy = mock_strategy

        result = dl.download("org/model")
        mock_strategy.download.assert_called_once_with(
            "org/model", False, tmp_path
        )

    def test_download_force_re_downloads(self, tmp_path):
        """force=True 即使已有模型也重新下载"""
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("weight")

        mock_strategy = MagicMock()
        mock_strategy.download.return_value = model_dir

        dl = ModelDownloader(cache_dir=tmp_path)
        dl._strategy = mock_strategy

        dl.download("org/model", force=True)
        mock_strategy.download.assert_called_once()

    def test_is_downloaded_detects_weights(self, tmp_path):
        """is_downloaded 检测权重文件"""
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        dl = ModelDownloader(cache_dir=tmp_path)
        assert dl.is_downloaded("org/model") is False

        (model_dir / "model.safetensors").write_text("weight")
        assert dl.is_downloaded("org/model") is True

    def test_list_downloaded_scans_recursively(self, tmp_path):
        """list_downloaded 递归扫描"""
        (tmp_path / "org" / "model").mkdir(parents=True)
        (tmp_path / "org" / "model" / "model.safetensors").write_text("w")

        dl = ModelDownloader(cache_dir=tmp_path)
        result = dl.list_downloaded()
        assert "org/model" in result
        assert result["org/model"] == tmp_path / "org" / "model"

    def test_remove_deletes_directory(self, tmp_path):
        """remove 删除模型目录"""
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("w")

        dl = ModelDownloader(cache_dir=tmp_path)
        assert dl.remove("org/model") is True
        assert not model_dir.exists()

    def test_model_dir_returns_path(self, tmp_path):
        """model_dir 返回计算路径"""
        dl = ModelDownloader(cache_dir=tmp_path)
        assert dl.model_dir("org/model") == tmp_path / "org" / "model"
```

- [ ] **Step 2: 运行测试验证全部失败（RED）**

```bash
cd "E:\Code\rag0709" && python -m pytest tests/unit/model/test_downloader.py -v 2>&1 | head -80
```

Expected: 全部 FAIL（因为策略类尚未实现）

- [ ] **Step 3: Commit**

```bash
git add tests/unit/model/test_downloader.py
git commit -m "test(model): add downloader strategy and ModelDownloader tests"
```

---

### Task 3: 实现 — 策略类 + ModelDownloader 重构

**Files:**
- Modify: `src/model/downloader.py`（全文重写）

**Interfaces:**
- Consumes: `ModelConfig.download_source: str`（Task 1），测试（Task 2）
- Produces: `HfStrategy`, `MsStrategy`, `AutoStrategy`, `DownloadStrategy` Protocol, `ModelDownloader`（重构后）

- [ ] **Step 1: 重写 `src/model/downloader.py`**

```python
"""
模型下载引擎 — 策略模式支持 HuggingFace / hf-mirror / ModelScope（魔搭）
"""

import shutil
import time
from pathlib import Path
from typing import Protocol

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from logger import logger

# ============================================================================
# 工具函数
# ============================================================================

# 模型权重文件扩展名
_WEIGHT_EXTS: frozenset = frozenset({
    ".safetensors", ".bin", ".onnx", ".msgpack", ".h5", ".ckpt", ".pt",
})


def _validate_model_id(model_id: str) -> str:
    """验证 model_id 不含路径穿越字符或绝对路径"""
    sanitized = model_id.replace("\\", "/")
    if sanitized != model_id:
        raise ValueError(f"model_id 包含非法字符: {model_id}")
    if ".." in sanitized.split("/"):
        raise ValueError(f"model_id 包含路径穿越: {model_id}")
    if sanitized.startswith("/"):
        raise ValueError(f"model_id 不能是绝对路径: {model_id}")
    return sanitized


# ============================================================================
# 下载策略协议
# ============================================================================

class DownloadStrategy(Protocol):
    """下载策略协议 — 只定义"如何下载"，不关心本地状态"""

    def download(self, model_id: str, force: bool,
                 cache_dir: Path, **kwargs) -> Path: ...


# ============================================================================
# HuggingFace 下载策略
# ============================================================================

class HfStrategy:
    """HuggingFace / hf-mirror 下载策略"""

    def __init__(self, endpoint: str | None = None,
                 token: str | None = None,
                 max_retries: int = 3):
        self._endpoint = endpoint
        self._token = token
        self._max_retries = max_retries

    def download(self, model_id: str, force: bool,
                 cache_dir: Path, **kwargs) -> Path:
        """通过 HuggingFace / hf-mirror 下载模型"""
        local_dir = cache_dir / model_id

        logger.info(f"开始下载模型 [HuggingFace]: {model_id} → {local_dir}")

        for attempt in range(1, self._max_retries + 1):
            try:
                sd_kwargs: dict = dict(
                    repo_id=model_id,
                    local_dir=str(local_dir),
                    token=self._token,
                )
                if self._endpoint:
                    sd_kwargs["endpoint"] = self._endpoint
                snapshot_download(**sd_kwargs)
                logger.info(f"模型下载完成 [HuggingFace]: {model_id}")
                return local_dir
            except RepositoryNotFoundError as e:
                raise ValueError(f"模型仓库不存在: {model_id}") from e
            except (HfHubHTTPError, OSError) as e:
                if attempt < self._max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        f"下载失败 (attempt {attempt}/{self._max_retries}), "
                        f"{wait}s 后重试: {e}"
                    )
                    time.sleep(wait)
                else:
                    # 清理不完整下载
                    if local_dir.is_dir() and not _has_weights(local_dir):
                        shutil.rmtree(local_dir, ignore_errors=True)
                    raise RuntimeError(
                        f"模型下载失败（已重试 {self._max_retries} 次）: {model_id}"
                    ) from e

        # 理论上不会到这里（最后一次重试要么成功要么抛异常）
        raise RuntimeError(
            f"模型下载失败（已重试 {self._max_retries} 次）: {model_id}"
        )


# ============================================================================
# ModelScope（魔搭）下载策略
# ============================================================================

class MsStrategy:
    """ModelScope（魔搭）下载策略 — 国内直连，免 token"""

    def __init__(self):
        try:
            from modelscope import snapshot_download as _ms_sd
            self._snapshot_download = _ms_sd
        except ImportError:
            raise RuntimeError(
                "modelscope 未安装。请运行: pip install modelscope"
            )

    def download(self, model_id: str, force: bool,
                 cache_dir: Path, **kwargs) -> Path:
        """通过 ModelScope 下载模型"""
        target_dir = cache_dir / model_id

        logger.info(f"开始下载模型 [ModelScope]: {model_id} → {target_dir}")

        try:
            tmp_dir = self._snapshot_download(
                model_id,
                cache_dir=str(cache_dir / ".modelscope_cache"),
            )
        except Exception as e:
            raise RuntimeError(
                f"ModelScope 下载失败: {model_id}\n"
                f"  错误: {e}\n"
                f"  提示: 检查 model_id 是否正确，或访问 https://modelscope.cn 确认模型存在"
            ) from e

        # 从 ModelScope 缓存目录复制到项目标准路径
        tmp_path = Path(tmp_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in tmp_path.iterdir():
            dst = target_dir / f.name
            if not dst.exists():
                if f.is_dir():
                    shutil.copytree(str(f), str(dst))
                else:
                    shutil.copy2(str(f), str(dst))

        # 验证权重文件
        if not _has_weights(target_dir):
            logger.warning(
                "未找到模型权重文件（.safetensors / .bin / .onnx 等），"
                "模型可能不完整"
            )

        logger.info(f"模型下载完成 [ModelScope]: {target_dir}")
        return target_dir


# ============================================================================
# 自动选择策略（MS 优先 → HF 回退）
# ============================================================================

class AutoStrategy:
    """自动选择 — 优先 ModelScope，失败回退 HuggingFace"""

    def __init__(self, ms: MsStrategy, hf: HfStrategy):
        self._ms = ms
        self._hf = hf

    def download(self, model_id: str, force: bool,
                 cache_dir: Path, **kwargs) -> Path:
        """先尝试 ModelScope，失败回退 HuggingFace"""
        try:
            return self._ms.download(model_id, force, cache_dir)
        except (RuntimeError, ImportError) as e:
            logger.info(
                f"ModelScope 下载失败，回退到 HuggingFace: {model_id} "
                f"（原因: {e}）"
            )
            return self._hf.download(model_id, force, cache_dir)


# ============================================================================
# 内部工具
# ============================================================================

def _has_weights(local_dir: Path) -> bool:
    """检查目录是否包含模型权重文件"""
    if not local_dir.is_dir():
        return False
    for f in local_dir.rglob("*"):
        if f.is_file() and f.suffix in _WEIGHT_EXTS:
            return True
    return False


# ============================================================================
# ModelDownloader
# ============================================================================

class ModelDownloader:
    """模型下载器 — 策略模式支持多下载源"""

    def __init__(self, cache_dir: Path, max_retries: int = 3,
                 hf_token: str | None = None,
                 endpoint: str | None = None,
                 download_source: str = "auto"):
        self._cache_dir = cache_dir
        self._max_retries = max_retries
        self._hf_token = hf_token
        self._endpoint = endpoint
        self._strategy = self._build_strategy(download_source)

    def _build_strategy(self, source: str) -> DownloadStrategy:
        """根据 download_source 组装对应的下载策略"""
        hf = HfStrategy(
            endpoint=self._endpoint,
            token=self._hf_token,
            max_retries=self._max_retries,
        )
        if source == "huggingface":
            return hf
        ms = MsStrategy()
        if source == "modelscope":
            return ms
        if source == "auto":
            return AutoStrategy(ms, hf)
        raise ValueError(
            f"不支持的 download_source: {source!r}，"
            f"可选: huggingface | modelscope | auto"
        )

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def model_dir(self, model_id: str) -> Path:
        return self._cache_dir / _validate_model_id(model_id)

    def download(self, model_id: str, force: bool = False) -> Path:
        """下载模型（委托给策略）"""
        model_id = _validate_model_id(model_id)
        local_dir = self._cache_dir / model_id

        if not force and self.is_downloaded(model_id):
            logger.info(f"模型已存在，跳过下载: {model_id} → {local_dir}")
            return local_dir

        return self._strategy.download(model_id, force, self._cache_dir)

    def is_downloaded(self, model_id: str) -> bool:
        """检查模型是否已下载（目录存在且包含模型权重文件）"""
        model_id = _validate_model_id(model_id)
        return _has_weights(self._cache_dir / model_id)

    def list_downloaded(self) -> dict[str, Path]:
        """列出所有已下载模型（只返回含权重文件的完整模型）"""
        result: dict[str, Path] = {}
        if not self._cache_dir.is_dir():
            return result

        def _scan(base: Path, prefix: str) -> None:
            for entry in base.iterdir():
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    current = f"{prefix}{entry.name}" if prefix else entry.name
                    if _has_weights(entry):
                        result[current] = entry
                    else:
                        _scan(entry, f"{current}/")

        _scan(self._cache_dir, "")
        return result

    def remove(self, model_id: str) -> bool:
        model_id = _validate_model_id(model_id)
        local_dir = self._cache_dir / model_id
        if local_dir.is_dir():
            shutil.rmtree(local_dir)
            logger.info(f"已删除模型目录: {local_dir}")
            return True
        return False


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    import sys
    from model import models
    models.download_all()
```

- [ ] **Step 2: 运行策略和 Downloader 测试**

```bash
cd "E:\Code\rag0709" && python -m pytest tests/unit/model/test_downloader.py -v 2>&1 | tail -40
```

Expected: 全部 PASS

- [ ] **Step 3: Commit**

```bash
git add src/model/downloader.py
git commit -m "refactor(model): integrate ModelScope download into ModelDownloader via strategy pattern"
```

---

### Task 4: ModelManager — 传递 `download_source` 参数

**Files:**
- Modify: `src/model/manager.py:79-84`

**Interfaces:**
- Consumes: `ModelDownloader(download_source=...)`（Task 3）
- Produces: 无新接口，内部集成

- [ ] **Step 1: 在 `manager.py` 中传递 `download_source`**

```python
# src/model/manager.py — _ensure_init 方法，修改 ModelDownloader 构造调用
            self._downloader = ModelDownloader(
                cache_dir=cache_dir,
                max_retries=cfg.max_retries,
                hf_token=token,
                endpoint=cfg.hf_endpoint,
                download_source=cfg.download_source,  # 新增
            )
```

- [ ] **Step 2: 验证 Manager 仍可正常初始化**

```bash
cd "E:\Code\rag0709" && python -c "from model import models; print('download_source:', models._downloader._strategy.__class__.__name__)"
```

Expected: `download_source: AutoStrategy`

- [ ] **Step 3: Commit**

```bash
git add src/model/manager.py
git commit -m "feat(model): pass download_source from config to ModelDownloader"
```

---

### Task 5: 导出 — 更新 `__init__.py`

**Files:**
- Modify: `src/model/__init__.py`

**Interfaces:**
- Consumes: `HfStrategy`, `MsStrategy`, `AutoStrategy`（Task 3）
- Produces: 对外导出策略类

- [ ] **Step 1: 更新 `__init__.py` 导出**

```python
# src/model/__init__.py — 在现有导入后新增策略类导出
from .downloader import (
    AutoStrategy,
    DownloadStrategy,
    HfStrategy,
    ModelDownloader,
    MsStrategy,
)
from .manager import ModelManager, models

__all__ = [
    "AutoStrategy",
    "DownloadStrategy",
    "HfStrategy",
    "ModelDownloader",
    "ModelManager",
    "MsStrategy",
    "__version__",
    "models",
]
```

- [ ] **Step 2: 验证导入**

```bash
cd "E:\Code\rag0709" && python -c "from model import HfStrategy, MsStrategy, AutoStrategy, DownloadStrategy; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add src/model/__init__.py
git commit -m "feat(model): export strategy classes from downloader module"
```

---

### Task 6: 文档 — 更新 README

**Files:**
- Modify: `src/model/README.md`

**Interfaces:**
- Consumes: 所有已实现功能（Task 1-5）
- Produces: 最新模块文档

- [ ] **Step 1: 更新 README 配置表和下载源说明**

在 `src/model/README.md` 的配置表格中新增一行，并在下载说明区域添加下载源切换说明。

在 `| max_retries | 3 | ...` 之后新增：

```markdown
| `download_source` | `auto` | 下载源: `huggingface` / `modelscope` / `auto` |
```

在"配置"段落之后，新增"下载源"小节：

```markdown
### 下载源切换

通过 `model.download_source` 配置选择下载源：

- **`huggingface`** — 使用 HuggingFace Hub（通过 `hf_endpoint` 可指定镜像站）
- **`modelscope`** — 使用 ModelScope（魔搭），国内直连免 token
- **`auto`**（默认）— 优先尝试 ModelScope，失败自动回退 HuggingFace

```python
# 编程方式切换（需在 ModelManager 初始化前设置）
from config import settings
settings.apply_overrides("model.download_source=modelscope")
```
```

同时更新文件结构表格，说明 `downloader.py` 包含策略类和下载器：

```markdown
| `downloader.py` | 下载策略（HfStrategy / MsStrategy / AutoStrategy）+ ModelDownloader |
```

- [ ] **Step 2: 确认 README 中不再引用已删除的 `download_from_modelscope()`**

```bash
cd "E:\Code\rag0709" && grep -n "download_from_modelscope" src/model/README.md || echo "No references — OK"
```

Expected: `No references — OK`

- [ ] **Step 3: Commit**

```bash
git add src/model/README.md
git commit -m "docs(model): update README with download_source config and strategy docs"
```

---

### Task 7: 最终验证 — 全量测试 + 配置一致性检查

**Files:**
- 无新建/修改

**Interfaces:**
- Consumes: 所有已完成任务

- [ ] **Step 1: 运行 downloader 测试**

```bash
cd "E:\Code\rag0709" && python -m pytest tests/unit/model/test_downloader.py -v
```

Expected: 全部 PASS

- [ ] **Step 2: 运行 model 模块所有测试**

```bash
cd "E:\Code\rag0709" && python -m pytest tests/unit/model/ -v
```

Expected: 全部 PASS（现有微调测试不受影响）

- [ ] **Step 3: 验证配置一致性**

```bash
cd "E:\Code\rag0709" && python -c "
from config import settings
from model import models
print('download_source config:', settings.model.download_source)
print('strategy type:', type(models._downloader._strategy).__name__)
assert settings.model.download_source == 'auto'
assert type(models._downloader._strategy).__name__ == 'AutoStrategy'
print('OK — config and runtime consistent')
"
```

Expected: `OK — config and runtime consistent`

- [ ] **Step 4: 验证 `download_from_modelscope` 已删除**

```bash
cd "E:\Code\rag0709" && python -c "from src.model.downloader import download_from_modelscope" 2>&1 || echo "ImportError — OK, function removed"
```

Expected: `ImportError — OK, function removed`

- [ ] **Step 5: Commit（如有残留变更）**

```bash
git status
```
