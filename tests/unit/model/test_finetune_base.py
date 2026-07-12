"""BaseTrainer 及结果模型测试"""

import tempfile
from pathlib import Path

from model.finetune.config import FinetuneConfig
from model.finetune.base import (
    BaseTrainer,
    FinetuneResult,
    FinetuneInfo,
)


class TestFinetuneResult:
    def test_default_construction(self):
        r = FinetuneResult(
            model_type="embedding",
            base_model="BAAI/bge-large-zh-v1.5",
            adapter_path=Path("/tmp/adapter"),
            output_name="test-v1",
        )
        assert r.model_type == "embedding"
        assert r.metrics == {}
        assert r.duration_seconds == 0.0


class TestFinetuneInfo:
    def test_construction(self):
        info = FinetuneInfo(
            name="my-lora",
            model_type="llm",
            base_model="Qwen/Qwen3-0.6B",
            adapter_path=Path("/tmp/adapter"),
            created_at="2026-07-12T00:00:00",
            metrics={"train_loss": 0.1},
            training_config={"epochs": 3},
        )
        assert info.name == "my-lora"
        assert info.model_type == "llm"


class TestBaseTrainerDeviceResolution:
    """测试 _resolve_device 逻辑"""

    def test_auto_device(self):
        # 用一个最小的具体子类来测试
        config = FinetuneConfig(device="auto")
        trainer = _DummyTrainer(config)
        device = trainer._resolve_device()
        assert str(device) in ("cuda", "cpu")

    def test_explicit_cpu(self):
        config = FinetuneConfig(device="cpu")
        trainer = _DummyTrainer(config)
        assert str(trainer._resolve_device()) == "cpu"


class TestBaseTrainerOutputDir:
    def test_output_dir_naming(self):
        output_dir = Path(tempfile.gettempdir()) / "finetuned"
        config = FinetuneConfig(output_dir=output_dir)
        trainer = _DummyTrainer(config)
        trainer._output_name = "my-test"
        out = trainer._get_output_dir()
        assert out == output_dir / "my-test"


class TestBaseTrainerMetadata:
    def test_save_and_load_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "test-adapter"
            adapter_dir.mkdir(parents=True)

            config = FinetuneConfig()
            trainer = _DummyTrainer(config)
            trainer._output_name = "test-adapter"
            trainer._metrics = {"train_loss": 0.15}

            result = FinetuneResult(
                model_type="embedding",
                base_model="test/model",
                adapter_path=adapter_dir,
                output_name="test-adapter",
                metrics={"train_loss": 0.15},
                duration_seconds=42.0,
            )
            trainer._save_metadata(result)

            loaded = BaseTrainer.load_metadata(adapter_dir)
            assert loaded is not None
            assert loaded["model_type"] == "embedding"
            assert loaded["base_model"] == "test/model"
            assert loaded["metrics"]["train_loss"] == 0.15

    def test_load_metadata_missing_file(self):
        assert BaseTrainer.load_metadata(Path("/nonexistent")) is None


class TestScanFinetuned:
    def test_scan_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = BaseTrainer.scan_finetuned(Path(tmp))
            assert result == {}

    def test_scan_with_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 创建两个适配器目录
            for name in ("adapter-a", "adapter-b"):
                adir = root / name
                adir.mkdir()
                config = FinetuneConfig()
                trainer = _DummyTrainer(config)
                trainer._output_name = name
                result = FinetuneResult(
                    model_type="llm", base_model="test/m",
                    adapter_path=adir, output_name=name,
                    metrics={}, duration_seconds=1.0,
                )
                trainer._save_metadata(result)

            scanned = BaseTrainer.scan_finetuned(root)
            assert len(scanned) == 2
            assert "adapter-a" in scanned
            assert scanned["adapter-a"].model_type == "llm"


# ============================================================
# 测试辅助：BaseTrainer 的具体最小实现
# ============================================================

class _DummyTrainer(BaseTrainer):
    """仅用于测试基类非抽象方法的虚拟子类"""

    model_type = "embedding"

    def __init__(self, config, base_model_id="test/model"):
        super().__init__(config, base_model_id)

    def load_data(self, data_path):
        return [], None

    def train(self, train_dataset, eval_dataset=None):
        out = self._get_output_dir()
        out.mkdir(parents=True, exist_ok=True)
        return out
