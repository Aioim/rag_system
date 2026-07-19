"""
HuggingFace 模型下载引擎 — 进度显示 + 断点续传 + 重试 + 错误处理
支持 HuggingFace / hf-mirror / ModelScope（魔塔）三种下载源
"""

import shutil
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from logger import logger


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


class ModelDownloader:
    """HuggingFace 模型下载器"""

    def __init__(self, cache_dir: Path, max_retries: int = 3,
                 hf_token: str | None = None,
                 endpoint: str | None = None):
        self._cache_dir = cache_dir
        self._max_retries = max_retries
        self._hf_token = hf_token
        self._endpoint = endpoint

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def model_dir(self, model_id: str) -> Path:
        return self._cache_dir / _validate_model_id(model_id)

    def download(self, model_id: str, force: bool = False) -> Path:
        """通过 HuggingFace / hf-mirror 下载模型"""
        model_id = _validate_model_id(model_id)
        local_dir = self._cache_dir / model_id

        if not force and self.is_downloaded(model_id):
            logger.info(f"模型已存在，跳过下载: {model_id} → {local_dir}")
            return local_dir

        logger.info(f"开始下载模型: {model_id} → {local_dir}")

        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs: dict = dict(
                    repo_id=model_id,
                    local_dir=str(local_dir),
                    token=self._hf_token,
                )
                if self._endpoint:
                    kwargs["endpoint"] = self._endpoint
                snapshot_download(**kwargs)
                logger.info(f"模型下载完成: {model_id}")
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
                    # 清理不完整下载（HF 可能下了 config 但无权重文件）
                    if local_dir.is_dir() and not self.is_downloaded(model_id):
                        shutil.rmtree(local_dir, ignore_errors=True)
                    raise RuntimeError(
                        f"模型下载失败（已重试 {self._max_retries} 次）: {model_id}"
                    ) from e

    def is_downloaded(self, model_id: str) -> bool:
        """检查模型是否已下载（目录存在且包含模型权重文件）"""
        model_id = _validate_model_id(model_id)
        local_dir = self._cache_dir / model_id
        if not local_dir.is_dir():
            return False
        for f in local_dir.rglob("*"):
            if f.is_file() and f.suffix in self._WEIGHT_EXTS:
                return True
        return False

    _WEIGHT_EXTS: frozenset = frozenset({".safetensors", ".bin", ".onnx", ".msgpack", ".h5", ".ckpt", ".pt"})

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
                    has_weights = any(
                        f.suffix in self._WEIGHT_EXTS
                        for f in entry.rglob("*") if f.is_file()
                    )
                    if has_weights:
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
# ModelScope（魔塔）下载支持 — 国内直连，免 token
# ============================================================================

_MODELSCOPE_AVAILABLE = False
try:
    from modelscope import snapshot_download as _ms_snapshot_download

    _MODELSCOPE_AVAILABLE = True
except ImportError:
    pass


def download_from_modelscope(model_id: str, cache_dir: str | Path = "models") -> Path:
    """通过 ModelScope（魔塔）下载模型 — 国内直连，无需 token

    使用示例：
        path = download_from_modelscope("BAAI/bge-large-zh-v1.5")
        path = download_from_modelscope("BAAI/bge-reranker-v2-m3", cache_dir="models")
    """
    if not _MODELSCOPE_AVAILABLE:
        raise RuntimeError(
            "modelscope 未安装。请运行: pip install modelscope"
        )

    model_id_safe = _validate_model_id(model_id)
    cache_dir = Path(cache_dir)
    target_dir = cache_dir / model_id_safe

    # 检查是否已有完整模型（含权重文件）
    weight_exts = {".safetensors", ".bin", ".onnx", ".msgpack", ".h5", ".ckpt", ".pt"}
    has_weights = (
        target_dir.is_dir()
        and any(
            f.suffix in weight_exts
            for f in target_dir.rglob("*")
            if f.is_file()
        )
    )
    if has_weights:
        logger.info(f"模型已存在（含权重），跳过下载: {model_id_safe} → {target_dir}")
        return target_dir

    logger.info(f"[ModelScope] 开始下载: {model_id_safe} → {target_dir}")

    try:
        tmp_dir = _ms_snapshot_download(
            model_id_safe,
            cache_dir=str(cache_dir / ".modelscope"),
        )
    except Exception as e:
        raise RuntimeError(
            f"ModelScope 下载失败: {model_id_safe}\n"
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
    weight_files = (
        list(target_dir.rglob("*.safetensors"))
        + list(target_dir.rglob("pytorch_model.bin"))
        + list(target_dir.rglob("*.onnx"))
    )
    if not weight_files:
        logger.warning(
            "⚠️  未找到模型权重文件（.safetensors / pytorch_model.bin / .onnx），"
            "模型可能不完整"
        )

    logger.info(f"[ModelScope] 下载完成: {target_dir}")
    return target_dir


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "modelscope":
        # python -m model.downloader modelscope BAAI/bge-large-zh-v1.5
        model = sys.argv[2] if len(sys.argv) > 2 else "BAAI/bge-large-zh-v1.5"
        download_from_modelscope(model)
    else:
        from model import models
        models.download_all()
