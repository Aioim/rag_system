"""
模型下载引擎 — 策略模式支持 HuggingFace / hf-mirror / ModelScope（魔搭）
"""

import re
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
    # Windows 绝对路径（如 C:/foo/bar）
    if re.match(r'^[a-zA-Z]:/', sanitized):
        raise ValueError(f"model_id 不能是 Windows 绝对路径: {model_id}")
    if sanitized != model_id and "\\" in model_id:
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
                    # 清理不完整下载（递归检查，权重可能在子目录）
                    if local_dir.is_dir() and not _has_weights(local_dir, recursive=True):
                        shutil.rmtree(local_dir, ignore_errors=True)
                    raise RuntimeError(
                        f"模型下载失败（已重试 {self._max_retries} 次）: {model_id}"
                    ) from e


# ============================================================================
# ModelScope（魔搭）下载策略
# ============================================================================

class MsStrategy:
    """ModelScope（魔搭）下载策略 — 国内直连，免 token"""

    def __init__(self):
        try:
            from modelscope import snapshot_download as _ms_sd
            self._snapshot_download = _ms_sd
        except ImportError as e:
            raise RuntimeError(
                "modelscope 未安装。请运行: pip install modelscope"
            ) from e

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

        # 验证权重文件（递归检查，ModelScope 可能嵌套存放）
        if not _has_weights(target_dir, recursive=True):
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
        except RuntimeError as e:
            logger.info(
                f"ModelScope 下载失败，回退到 HuggingFace: {model_id} "
                f"（原因: {e}）"
            )
            return self._hf.download(model_id, force, cache_dir)


# ============================================================================
# 内部工具
# ============================================================================

def _has_weights(local_dir: Path, *, recursive: bool = False) -> bool:
    """检查目录是否包含模型权重文件。

    Args:
        local_dir: 待检查目录
        recursive: False — 仅检查直接子文件（list_downloaded 的 _scan 用此区分
                   命名空间目录与模型目录）；True — 递归扫描任意深度
                   （is_downloaded / 下载完整性验证等场景）
    """
    if not local_dir.is_dir():
        return False
    iterator = local_dir.rglob("*") if recursive else local_dir.glob("*")
    return any(f.is_file() and f.suffix in _WEIGHT_EXTS for f in iterator)


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

        # 延迟构造 MsStrategy — 避免 auto 模式在 modelscope 未安装时崩溃
        ms: MsStrategy | None = None
        try:
            ms = MsStrategy()
        except RuntimeError:
            if source == "modelscope":
                raise  # 明确指定 modelscope 但未安装 → 直接报错

        if source == "modelscope":
            return ms  # type: ignore[return-value]  # try 块成功时 ms 非 None
        if source == "auto":
            if ms is not None:
                return AutoStrategy(ms, hf)
            logger.info("modelscope 未安装，auto 模式仅使用 HuggingFace")
            return hf
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
        """检查模型是否已下载（目录存在且在任意深度包含模型权重文件）"""
        model_id = _validate_model_id(model_id)
        return _has_weights(self._cache_dir / model_id, recursive=True)

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
    from model import models
    models.download_all()
