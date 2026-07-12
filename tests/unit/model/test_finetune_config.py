"""FinetuneConfig 配置模型测试"""

import tempfile
from pathlib import Path

import pytest

from model.finetune.config import (
    FinetuneConfig,
    LoRAConfig,
    TrainingConfig,
    DistillationConfig,
    get_finetune_config,
)


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r == 8
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.1
        assert cfg.target_modules is None

    def test_custom_values(self):
        cfg = LoRAConfig(r=16, lora_alpha=64, lora_dropout=0.2,
                         target_modules=["q_proj", "v_proj"])
        assert cfg.r == 16
        assert cfg.target_modules == ["q_proj", "v_proj"]

    def test_r_out_of_range_raises(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            LoRAConfig(r=0)


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 3
        assert cfg.learning_rate == 2.0e-4
        assert cfg.batch_size == 8

    def test_epochs_must_be_positive(self):
        with pytest.raises(Exception):
            TrainingConfig(epochs=0)


class TestDistillationConfig:
    def test_defaults(self):
        cfg = DistillationConfig()
        assert cfg.temperature == 2.0
        assert cfg.alpha == 0.5

    def test_alpha_clamped(self):
        with pytest.raises(Exception):
            DistillationConfig(alpha=1.5)


class TestFinetuneConfig:
    def test_default_construction(self):
        cfg = FinetuneConfig()
        assert cfg.device == "auto"
        assert cfg.training.epochs == 3
        assert cfg.lora.r == 8
        assert cfg.distillation.alpha == 0.5

    def test_resolve_output_dir_relative(self):
        cfg = FinetuneConfig(output_dir=Path("models/finetuned"))
        resolved = cfg.resolve_output_dir(Path("/project"))
        assert resolved == Path("/project/models/finetuned")

    def test_resolve_output_dir_absolute(self):
        cfg = FinetuneConfig(output_dir=Path("/absolute/path"))
        resolved = cfg.resolve_output_dir(Path("/project"))
        assert resolved == Path("/absolute/path")

    def test_resolve_data_dir_relative(self):
        cfg = FinetuneConfig(data_dir=Path("data/finetune"))
        resolved = cfg.resolve_data_dir(Path("/project"))
        assert resolved == Path("/project/data/finetune")

    def test_from_yaml_returns_defaults_when_no_settings(self):
        """无 config.settings 时返回默认配置"""
        cfg = FinetuneConfig.from_yaml(settings_module=None)
        assert cfg.device == "auto"
        assert cfg.training.epochs == 3
