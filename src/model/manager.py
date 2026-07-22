"""
模型管理器单例 — 统一下载 / 查询 / 删除 embedding、rerank、LLM 模型
"""

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from .finetune.base import BaseTrainer, FinetuneInfo, FinetuneResult
    from .finetune.config import FinetuneConfig

from config import settings
from logger import logger

from .downloader import ModelDownloader

# "reranker" → "rerank" 等类型别名映射（默认配置中以短名称存储）
from .finetune.aliases import _MODEL_TYPE_ALIASES


class ModelManager:
    """模型管理器 — 线程安全单例

    支持三种模型类型：embedding / rerank / llm
    使用配置中的 default_models 映射类型到 HuggingFace repo_id。

    使用示例：
        from model import models

        # 下载所有默认模型
        models.download_all()

        # 按类型下载
        models.download("embedding")

        # 按 HuggingFace repo_id 下载
        models.download("BAAI/bge-large-zh-v1.5")

        # 查询
        models.get_path("embedding")             # → Path | None
        models.is_downloaded("rerank")           # → bool
        models.status()                          # → dict[str, bool]
        models.list_downloaded()                 # → dict[str, Path]
    """

    _instance: ClassVar[Optional["ModelManager"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._downloader: ModelDownloader | None = None
                    instance._defaults: dict[str, str] = {}
                    instance._cache_dir: Path | None = None
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def _ensure_init(self) -> None:
        """延迟初始化，首次访问时自动触发"""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            cfg = settings.model
            token = os.getenv(cfg.hf_token_env) or None

            # 解析缓存目录：相对路径相对于 PROJECT_ROOT
            cache_dir = cfg.cache_dir
            if not cache_dir.is_absolute():
                from config.path import PROJECT_ROOT
                cache_dir = PROJECT_ROOT / cache_dir

            self._downloader = ModelDownloader(
                cache_dir=cache_dir,
                max_retries=cfg.max_retries,
                hf_token=token,
                endpoint=cfg.hf_endpoint,
                download_source=cfg.download_source,
            )
            self._defaults = dict(cfg.default_models)
            self._cache_dir = cache_dir
            # 确保缓存目录存在
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    # ========================================================================
    # 公共 API
    # ========================================================================

    def download_all(self) -> list[Path]:
        """下载所有默认模型，返回已下载的路径列表"""
        self._ensure_init()
        paths: list[Path] = []
        errors: dict[str, str] = {}
        for model_type, model_id in self._defaults.items():
            if not model_id:
                continue
            try:
                path = self._download_by_id(model_id)
                paths.append(path)
            except (ValueError, RuntimeError, OSError) as e:
                logger.error(f"下载失败 [{model_type}] {model_id}: {e}")
                errors[model_type] = str(e)
        if errors:
            logger.error(f"部分模型下载失败: {errors}")
            raise RuntimeError(
                f"部分模型下载失败: {errors}"
            )
        return paths

    def download(self, type_or_id: str) -> Path:
        """下载指定模型。

        - 如果是类型名（embedding / rerank / llm），下载对应的默认模型
        - 如果包含 '/'（HuggingFace repo_id），直接下载
        """
        self._ensure_init()
        model_id = self._resolve_model_id(type_or_id)
        return self._download_by_id(model_id)

    def get_default_model_id(self, model_type: str) -> str | None:
        """获取指定类型的默认模型 ID（不触发初始化副作用）"""
        self._ensure_init()
        return self._defaults.get(model_type)

    def get_path(self, type_or_id: str) -> Path | None:
        """获取模型本地路径（不触发下载）"""
        self._ensure_init()
        model_id = self._resolve_model_id(type_or_id)
        if self._downloader.is_downloaded(model_id):
            return self._downloader.model_dir(model_id)
        return None

    def is_downloaded(self, type_or_id: str) -> bool:
        """检查模型是否已下载"""
        self._ensure_init()
        model_id = self._resolve_model_id(type_or_id)
        return self._downloader.is_downloaded(model_id)

    def list_downloaded(self) -> dict[str, Path]:
        """列出所有已下载的模型，返回 {model_id: local_path}"""
        self._ensure_init()
        return self._downloader.list_downloaded()

    def status(self) -> dict[str, bool]:
        """各类型的下载状态：{model_type: is_downloaded}"""
        self._ensure_init()
        downloaded = self._downloader.list_downloaded()
        return {
            mt: mid in downloaded
            for mt, mid in self._defaults.items()
            if mid
        }

    def remove(self, type_or_id: str) -> bool:
        """删除已下载的模型目录"""
        self._ensure_init()
        model_id = self._resolve_model_id(type_or_id)
        return self._downloader.remove(model_id)

    def remove_all(self) -> int:
        """删除所有已下载模型，返回删除数量"""
        self._ensure_init()
        count = 0
        for model_id in list(self._downloader.list_downloaded().keys()):
            if self._downloader.remove(model_id):
                count += 1
        return count

    # ========================================================================
    # 微调 API
    # ========================================================================

    def finetune(
        self,
        model_type: str,
        data_path: str | Path,
        output_name: str | None = None,
        teacher: str | None = None,
        config: Optional["FinetuneConfig"] = None,
        **overrides,
    ) -> "FinetuneResult":
        """微调指定类型的模型。

        Args:
            model_type: "embedding" | "reranker" | "llm"
            data_path: JSONL 训练数据路径
            output_name: 适配器名称（默认自动生成）
            teacher: 蒸馏教师模型 ID（仅 llm）
            config: 微调配置（默认从 YAML 加载）
            **overrides: 覆盖训练参数，如 epochs=5, batch_size=4

        Returns:
            FinetuneResult with adapter_path, metrics, etc.
        """
        from .finetune.config import get_finetune_config
        from .finetune.embedding_trainer import EmbeddingTrainer
        from .finetune.llm_trainer import LLMTrainer
        from .finetune.reranker_trainer import RerankerTrainer

        self._ensure_init()

        # 解析配置（有 overrides 时基于 YAML 配置深拷贝，避免污染缓存单例）
        if config is not None:
            cfg = config
        elif overrides:
            cfg = get_finetune_config().model_copy(deep=True)
        else:
            cfg = get_finetune_config()

        # 解析类型别名（"reranker" → "rerank"）
        resolved_type = _MODEL_TYPE_ALIASES.get(model_type, model_type)

        # 应用 overrides（通过 model_copy 创建新对象，保持 Pydantic 不可变语义）
        training_updates: dict = {}
        lora_updates: dict = {}
        distill_updates: dict = {}
        for key, value in overrides.items():
            if hasattr(cfg.training, key):
                training_updates[key] = value
            elif hasattr(cfg.lora, key):
                lora_updates[key] = value
            elif hasattr(cfg.distillation, key):
                distill_updates[key] = value
        if training_updates:
            cfg.training = cfg.training.model_copy(update=training_updates)
        if lora_updates:
            cfg.lora = cfg.lora.model_copy(update=lora_updates)
        if distill_updates:
            cfg.distillation = cfg.distillation.model_copy(update=distill_updates)

        # 解析基座模型
        if resolved_type not in self._defaults:
            raise ValueError(
                f"不支持的模型类型: {model_type}（解析后: {resolved_type}），"
                f"可选: {list(self._defaults.keys())}，"
                f"别名: {list(_MODEL_TYPE_ALIASES.keys())}"
            )
        base_model_id = self._defaults[resolved_type]

        # 选择 Trainer
        data_path_obj = Path(data_path) if not isinstance(data_path, Path) else data_path
        trainer_classes: dict[str, type] = {
            "embedding": EmbeddingTrainer,
            "rerank": RerankerTrainer,
            "llm": LLMTrainer,
        }

        trainer_cls = trainer_classes[resolved_type]
        trainer: BaseTrainer
        if resolved_type == "llm" and teacher:
            trainer = LLMTrainer(cfg, base_model_id, teacher_model=teacher)
        else:
            trainer = trainer_cls(cfg, base_model_id)

        logger.info(
            f"开始微调 [{model_type}] base={base_model_id} data={data_path_obj}"
            + (f" teacher={teacher}" if teacher else "")
        )

        return trainer.run(data_path_obj, output_name=output_name)

    def list_finetuned(self) -> dict[str, "FinetuneInfo"]:
        """列出所有已微调的适配器，返回 {name: FinetuneInfo}"""
        self._ensure_init()
        from config.path import PROJECT_ROOT

        from .finetune.base import BaseTrainer
        from .finetune.config import get_finetune_config

        cfg = get_finetune_config()
        output_dir = cfg.resolve_output_dir(PROJECT_ROOT)
        return BaseTrainer.scan_finetuned(output_dir)

    def get_finetuned_path(self, name: str) -> Path | None:
        """获取指定适配器的本地路径"""
        adapters = self.list_finetuned()
        info = adapters.get(name)
        return info.adapter_path if info else None

    def remove_finetuned(self, name: str) -> bool:
        """删除指定适配器，成功返回 True"""
        import shutil
        path = self.get_finetuned_path(name)
        if path is None:
            return False
        shutil.rmtree(path)
        logger.info(f"已删除微调适配器: {name} ({path})")
        return True

    # ========================================================================
    # 推理 API（委托至 inference 模块）
    # ========================================================================

    def encode(self, texts: str | list[str], **kwargs) -> "np.ndarray":
        """对文本进行 embedding 编码。

        详见 ``inference.encode()``。
        """
        from . import inference as _inference
        return _inference.encode(texts, **kwargs)

    def rerank(self, query: str, documents: list[str], **kwargs) -> list[dict]:
        """对查询与候选文档进行相关性排序。

        详见 ``inference.rerank()``。
        """
        from . import inference as _inference
        return _inference.rerank(query, documents, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """LLM 文本生成（预留接口，当前未实现）。

        详见 ``inference.generate()``。
        """
        from . import inference as _inference
        return _inference.generate(prompt, **kwargs)

    @property
    def embedding_model(self):
        """获取 SentenceTransformer 实例（懒加载 + 双检锁）"""
        from . import inference as _inference
        return _inference._get_embedding_model()

    @property
    def cross_encoder(self):
        """获取 CrossEncoder 实例（懒加载 + 双检锁）"""
        from . import inference as _inference
        return _inference._get_cross_encoder()

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _resolve_model_id(self, type_or_id: str) -> str:
        """将类型名解析为 model_id"""
        if type_or_id in self._defaults:
            model_id = self._defaults[type_or_id]
            if not model_id:
                raise ValueError(
                    f"模型类型 '{type_or_id}' 未配置默认 model_id"
                )
            return model_id
        if "/" in type_or_id:
            return type_or_id
        raise ValueError(
            f"无法解析 '{type_or_id}'：不是有效的模型类型 {list(self._defaults.keys())} "
            f"或 HuggingFace repo_id"
        )

    def _download_by_id(self, model_id: str) -> Path:
        """按 model_id 下载模型"""
        return self._downloader.download(model_id)


# ========================================================================
# 全局单例导出
# ========================================================================

models = ModelManager()
