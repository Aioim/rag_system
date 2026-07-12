"""
微调配置模型 — Pydantic v2 + YAML 配置加载
"""

from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """LoRA 适配器参数"""

    r: int = Field(default=8, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha 缩放因子")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: Optional[list[str]] = Field(
        default=None, description="目标模块名列表，None 则按模型类型自动推断"
    )


class TrainingConfig(BaseModel):
    """训练超参数"""

    epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=2.0e-4, gt=0.0)
    batch_size: int = Field(default=8, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_seq_length: int = Field(default=512, ge=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    save_steps: int = Field(default=500, ge=1)
    logging_steps: int = Field(default=50, ge=1)


class DistillationConfig(BaseModel):
    """蒸馏参数（仅 LLM）"""

    temperature: float = Field(default=2.0, gt=0.0, description="软化教师分布的温度")
    alpha: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="硬标签权重（0=纯蒸馏，1=纯SFT）"
    )


class FinetuneConfig(BaseModel):
    """微调总配置"""

    output_dir: Path = Field(default=Path("models/finetuned"), description="LoRA 适配器输出目录")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    data_dir: Path = Field(default=Path("data/finetune"), description="训练数据默认目录")
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)

    def resolve_output_dir(self, project_root: Path) -> Path:
        """将相对路径 output_dir 解析为绝对路径"""
        if self.output_dir.is_absolute():
            return self.output_dir
        return project_root / self.output_dir

    def resolve_data_dir(self, project_root: Path) -> Path:
        """将相对路径 data_dir 解析为绝对路径"""
        if self.data_dir.is_absolute():
            return self.data_dir
        return project_root / self.data_dir

    @classmethod
    def from_yaml(cls, settings_module=None) -> "FinetuneConfig":
        """从 config.settings YAML 加载配置，不存在则使用默认值。

        通过 config.settings.get() 逐项读取，
        将 defaults.yaml 中 finetune: 段映射为 Pydantic 对象。
        """
        try:
            from config import settings
        except ImportError:
            return cls()

        try:
            finetune_cfg = settings.get("finetune") or {}
        except Exception:
            finetune_cfg = {}

        if not finetune_cfg:
            return cls()

        # 解析嵌套结构
        training_raw = finetune_cfg.get("training", {})
        lora_raw = finetune_cfg.get("lora", {})
        distill_raw = finetune_cfg.get("distillation", {})

        return cls(
            output_dir=Path(finetune_cfg.get("output_dir", "models/finetuned")),
            device=finetune_cfg.get("device", "auto"),
            data_dir=Path(finetune_cfg.get("data_dir", "data/finetune")),
            training=TrainingConfig(**training_raw) if training_raw else TrainingConfig(),
            lora=LoRAConfig(**lora_raw) if lora_raw else LoRAConfig(),
            distillation=DistillationConfig(**distill_raw) if distill_raw else DistillationConfig(),
        )


# 缓存
_config_cache: Optional[FinetuneConfig] = None


def get_finetune_config() -> FinetuneConfig:
    """获取微调配置单例（带缓存，首次调用从 YAML 加载）"""
    global _config_cache
    if _config_cache is None:
        _config_cache = FinetuneConfig.from_yaml()
    return _config_cache


def reload_finetune_config() -> FinetuneConfig:
    """强制重新加载微调配置"""
    global _config_cache
    _config_cache = FinetuneConfig.from_yaml()
    return _config_cache
