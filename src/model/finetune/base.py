"""
微调基类 — BaseTrainer ABC + FinetuneResult + FinetuneInfo + metadata 管理
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml


@dataclass
class FinetuneResult:
    """单次微调训练的结果"""

    model_type: str              # "embedding" | "reranker" | "llm"
    base_model: str              # 基座模型 HuggingFace repo_id
    adapter_path: Path           # LoRA 适配器保存路径
    output_name: str             # 适配器名称
    metrics: dict = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class FinetuneInfo:
    """已存储适配器的元信息（从 metadata.yaml 反序列化）"""

    name: str
    model_type: str
    base_model: str
    adapter_path: Path
    created_at: str
    metrics: dict = field(default_factory=dict)
    training_config: dict = field(default_factory=dict)


class BaseTrainer(ABC):
    """模型微调抽象基类

    子类必须实现:
      - load_data(data_path) -> (train_dataset, eval_dataset|None)
      - train(train_dataset, eval_dataset) -> Path (adapter save path)

    子类应设置:
      - self.model_type: str
    """

    model_type: str = ""  # 子类覆盖

    def __init__(self, config, base_model_id: str):
        from .config import FinetuneConfig
        self.config: FinetuneConfig = config
        self.base_model_id = base_model_id
        self._output_name: str | None = None
        self._start_time: float = 0.0
        self._metrics: dict = {}

    # ---- 子类必须实现 ----

    @abstractmethod
    def load_data(self, data_path: Path, records: list[dict] | None = None):
        """加载训练数据，返回 (train_dataset, eval_dataset)。

        Args:
            data_path: JSONL 数据文件路径
            records: 已解析的记录列表（由 run() 提供以避免重复解析）。
                     为 None 时从 data_path 自行加载（兼容直接调用 load_data 的场景）。
        """
        ...

    @abstractmethod
    def train(self, train_dataset, eval_dataset=None) -> Path:
        """执行训练，返回 LoRA 适配器保存目录的 Path"""
        ...

    # ---- 模板方法 ----

    @staticmethod
    def _validate_output_name(name: str) -> str:
        """校验 output_name 不含路径穿越字符"""
        import re
        if not re.match(r'^[\w\-]+$', name):
            raise ValueError(f"output_name 包含非法字符: {name!r}，仅允许字母、数字、下划线、连字符")
        return name

    def run(self, data_path: Path, output_name: str | None = None) -> FinetuneResult:
        """完整训练流程：验证 → 加载 → 训练 → 保存元数据 → 返回结果"""
        self._output_name = self._validate_output_name(
            output_name or self._generate_default_name()
        )
        self._start_time = time.time()

        # 1. 加载 + 校验数据（只解析一次 JSONL，load_data 复用）
        from .data import load_jsonl
        records = load_jsonl(data_path)
        self._validate_records(records)

        # 2. 加载数据（复用已解析的 records）
        train_ds, eval_ds = self.load_data(data_path, records)

        # 3. 训练
        adapter_path = self.train(train_ds, eval_ds)

        # 4. 组装结果
        self._metrics["duration_seconds"] = time.time() - self._start_time
        result = FinetuneResult(
            model_type=self.model_type,
            base_model=self.base_model_id,
            adapter_path=adapter_path,
            output_name=self._output_name,
            metrics=self._metrics,
            duration_seconds=self._metrics["duration_seconds"],
        )

        # 5. 保存元数据
        self._save_metadata(result)

        return result

    def _generate_default_name(self) -> str:
        """生成默认适配器名称: {model_type}_{YYYYMMDD_HHMMSS}"""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return f"{self.model_type}_{timestamp}"

    def _validate_records(self, records: list[dict]) -> None:
        """使用 data.py 中的校验器检查数据格式"""
        from .data import VALIDATORS

        if self.model_type not in VALIDATORS:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        VALIDATORS[self.model_type](records)

    def _resolve_device(self) -> torch.device:
        """解析 device 配置: auto → cuda(如果可用) → cpu"""
        device_str = self.config.device
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _get_output_dir(self) -> Path:
        """返回适配器输出目录"""
        if self._output_name is None:
            raise RuntimeError(
                "_output_name 未设置。请通过 run() 方法调用训练，"
                "或在调用 train() 之前手动设置 _output_name。"
            )
        from config.path import PROJECT_ROOT
        return self.config.resolve_output_dir(PROJECT_ROOT) / self._output_name

    def _save_metadata(self, result: FinetuneResult) -> None:
        """保存 metadata.yaml 到适配器目录"""
        meta = {
            "model_type": result.model_type,
            "base_model": result.base_model,
            "output_name": result.output_name,
            "created_at": datetime.now(UTC).isoformat(),
            "metrics": result.metrics,
            "training_config": {
                "epochs": self.config.training.epochs,
                "learning_rate": self.config.training.learning_rate,
                "batch_size": self.config.training.batch_size,
                "lora_r": self.config.lora.r,
                "lora_alpha": self.config.lora.lora_alpha,
            },
        }
        meta_path = result.adapter_path / "metadata.yaml"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = meta_path.with_suffix(".yaml.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)
            os.replace(tmp_path, meta_path)
        except (OSError, yaml.YAMLError):
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    # ---- 静态工具方法（供 ModelManager 使用） ----

    @staticmethod
    def load_metadata(adapter_dir: Path) -> dict | None:
        """读取适配器目录中的 metadata.yaml"""
        meta_path = adapter_dir / "metadata.yaml"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            from logger import logger
            logger.warning(f"无法读取元数据文件 {meta_path}: {e}")
            return None

    @staticmethod
    def scan_finetuned(output_dir: Path) -> dict[str, FinetuneInfo]:
        """扫描 output_dir，返回 {name: FinetuneInfo}"""
        result: dict[str, FinetuneInfo] = {}
        if not output_dir.is_dir():
            return result
        for entry in sorted(output_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            meta = BaseTrainer.load_metadata(entry)
            if meta is None:
                continue
            name = meta.get("output_name", entry.name)
            result[name] = FinetuneInfo(
                name=name,
                model_type=meta.get("model_type", "unknown"),
                base_model=meta.get("base_model", ""),
                adapter_path=entry,
                created_at=meta.get("created_at", ""),
                metrics=meta.get("metrics", {}),
                training_config=meta.get("training_config", {}),
            )
        return result
