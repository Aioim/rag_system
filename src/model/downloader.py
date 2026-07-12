"""
HuggingFace 模型下载引擎 — 进度显示 + 断点续传 + 重试 + 错误处理
"""

import time
import shutil
from pathlib import Path
from typing import Optional, Dict

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
    """HuggingFace 模型下载器

    封装 huggingface_hub.snapshot_download，提供：
    - 断点续传（resume_download=True，内置支持）
    - 失败重试（指数退避：1s / 2s / 4s）
    - 进度条（tqdm，huggingface_hub 内置）
    - 路径穿越防护

    使用示例：
        dl = ModelDownloader(Path("models"))
        path = dl.download("BAAI/bge-large-zh-v1.5")
    """

    def __init__(self, cache_dir: Path, max_retries: int = 3,
                 hf_token: Optional[str] = None,
                 endpoint: Optional[str] = None):
        self._cache_dir = cache_dir
        self._max_retries = max_retries
        self._hf_token = hf_token
        self._endpoint = endpoint

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def model_dir(self, model_id: str) -> Path:
        """返回模型的本地目录路径（安全验证后拼接）"""
        return self._cache_dir / _validate_model_id(model_id)

    def download(self, model_id: str, force: bool = False) -> Path:
        """下载模型，返回本地路径。

        如果模型已存在则跳过（除非 force=True）。
        使用 snapshot_download 实现断点续传。
        """
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
            except RepositoryNotFoundError:
                raise ValueError(f"模型仓库不存在: {model_id}")
            except (HfHubHTTPError, OSError) as e:
                if attempt < self._max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        f"下载失败 (attempt {attempt}/{self._max_retries}), "
                        f"{wait}s 后重试: {e}"
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"模型下载失败（已重试 {self._max_retries} 次）: {model_id}"
                    ) from e

    def is_downloaded(self, model_id: str) -> bool:
        """检查模型是否已下载（目录存在且有非隐藏的模型文件）"""
        model_id = _validate_model_id(model_id)
        local_dir = self._cache_dir / model_id
        if not local_dir.is_dir():
            return False
        for f in local_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                return True
        return False

    def list_downloaded(self) -> Dict[str, Path]:
        """列出所有已下载模型，返回 {model_id: local_path}

        递归扫描缓存目录以正确处理层级化的 model_id（如 BAAI/bge-large-zh-v1.5）。
        """
        result: Dict[str, Path] = {}
        if not self._cache_dir.is_dir():
            return result

        def _scan(base: Path, prefix: str) -> None:
            for entry in base.iterdir():
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    current = f"{prefix}{entry.name}" if prefix else entry.name
                    # 检查是否包含模型文件（非目录的文件）
                    files = [f for f in entry.iterdir()
                             if f.is_file() and not f.name.startswith(".")]
                    if files:
                        result[current] = entry
                    else:
                        # 无模型文件则继续向下扫描
                        _scan(entry, f"{current}/")

        _scan(self._cache_dir, "")
        return result

    def remove(self, model_id: str) -> bool:
        """删除已下载的模型目录"""
        model_id = _validate_model_id(model_id)
        local_dir = self._cache_dir / model_id
        if local_dir.is_dir():
            shutil.rmtree(local_dir)
            logger.info(f"已删除模型目录: {local_dir}")
            return True
        return False


if __name__ == "__main__":
    from model import models
    models.download_all()