"""ModelManager 微调 API 测试"""

import tempfile
import json
from pathlib import Path

import pytest

from model import models
from model.finetune.config import FinetuneConfig


def _make_triplet_jsonl(path: Path, count: int = 10) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({
                "query": f"q{i}", "positive": f"p{i}", "negative": f"n{i}",
            }, ensure_ascii=False) + "\n")


class TestFinetuneAPI:
    """测试 ModelManager 新增的微调 API（不触发实际训练）"""

    def test_list_finetuned_empty(self):
        # 使用临时目录作为 output_dir
        with tempfile.TemporaryDirectory() as tmp:
            config = FinetuneConfig(output_dir=Path(tmp))
            # 由于 list_finetuned 读取 get_finetune_config().output_dir，
            # 这里验证空目录返回 {}
            # 不直接依赖 models.list_finetuned() 因为它会读取全局配置
            from model.finetune.base import BaseTrainer
            result = BaseTrainer.scan_finetuned(Path(tmp))
            assert result == {}

    def test_invalid_model_type_raises(self, monkeypatch):
        """models.finetune() 对无效类型应报错"""
        # 使用 monkeypatch 设置默认值，避免触发真实初始化副作用，
        # 也不污染单例的 _initialized 状态到后续测试
        monkeypatch.setattr(models, '_defaults', {"embedding": "BAAI/bge-large-zh-v1.5"}, raising=False)
        monkeypatch.setattr(models, '_initialized', True, raising=False)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "test.jsonl"
            _make_triplet_jsonl(data_path, count=3)

            # 在未安装模型的环境下，先检查类型校验
            # 不是测试实际训练
            try:
                models.finetune("invalid_type", str(data_path))
                assert False, "应该抛出 ValueError"
            except ValueError as e:
                assert "不支持的模型类型" in str(e)

    def test_get_finetuned_path_not_found(self):
        """不存在的适配器返回 None"""
        # 直接测逻辑：scan 空目录 → 找不到
        with tempfile.TemporaryDirectory() as tmp:
            from model.finetune.base import BaseTrainer
            scanned = BaseTrainer.scan_finetuned(Path(tmp))
            assert scanned.get("nonexistent") is None

    def test_aliases_resolve_reranker(self):
        """"reranker" 应被别名解析为 "rerank" 以匹配配置"""
        from model.manager import _MODEL_TYPE_ALIASES
        assert _MODEL_TYPE_ALIASES["reranker"] == "rerank"

    def test_overrides_reaches_lora_and_distillation(self):
        """**overrides 应能写入 cfg.lora 和 cfg.distillation"""
        from model.finetune.config import FinetuneConfig

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "test.jsonl"
            _make_triplet_jsonl(data_path, count=3)

            config = FinetuneConfig(output_dir=tmp)

            # 使用一个必定不在 _defaults 中的类型名，
            # 确保 finetune() 在 overrides 循环之后因类型校验而失败，
            # 不会进入实际创建 Trainer 的步骤
            try:
                models.finetune(
                    "definitely_not_a_valid_type",
                    str(data_path),
                    config=config,
                    r=16,
                    alpha=0.3,
                )
            except ValueError:
                pass

            assert config.lora.r == 16, f"lora.r 应为 16，实际: {config.lora.r}"
            assert config.distillation.alpha == 0.3, (
                f"distillation.alpha 应为 0.3，实际: {config.distillation.alpha}"
            )

    def test_overrides_preserve_yaml_config(self, monkeypatch):
        """**overrides 应叠加在 YAML 配置之上，而非丢弃 YAML 退回全默认值（复现审查发现）"""

        class _CaptureTrainer:
            captured_cfg = None

            def __init__(self, cfg, base_model_id, **kwargs):
                _CaptureTrainer.captured_cfg = cfg
                raise RuntimeError("stop-before-training")

        yaml_cfg = FinetuneConfig()
        yaml_cfg.training.epochs = 99  # 模拟 YAML 自定义值（代码默认为 3）

        # 确保 models 单例在测试环境中正确初始化，避免受其他测试污染
        monkeypatch.setattr(models, '_defaults', {"embedding": "BAAI/bge-large-zh-v1.5"}, raising=False)
        monkeypatch.setattr(models, '_initialized', True, raising=False)

        monkeypatch.setattr(
            "model.finetune.config.get_finetune_config", lambda: yaml_cfg
        )
        monkeypatch.setattr(
            "model.finetune.embedding_trainer.EmbeddingTrainer", _CaptureTrainer
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "test.jsonl"
            _make_triplet_jsonl(data_path, count=3)

            with pytest.raises(RuntimeError, match="stop-before-training"):
                models.finetune("embedding", str(data_path), batch_size=4)

        cfg = _CaptureTrainer.captured_cfg
        assert cfg is not None
        assert cfg.training.batch_size == 4, "override 应生效"
        assert cfg.training.epochs == 99, "YAML 自定义值应保留，而非退回默认值"
        # 不污染缓存单例：overrides 写入的是深拷贝
        assert cfg is not yaml_cfg
        assert yaml_cfg.training.batch_size == 8
