"""EmbeddingTrainer 单元测试"""

import tempfile
import json
from pathlib import Path

import pytest

from model.finetune.config import FinetuneConfig
from model.finetune.embedding_trainer import EmbeddingTrainer


def _make_triplet_jsonl(path: Path, count: int = 10) -> None:
    """生成测试用三元组 JSONL"""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({
                "query": f"问题{i}",
                "positive": f"正确答案{i}",
                "negative": f"错误答案{i}",
            }, ensure_ascii=False) + "\n")


class TestEmbeddingTrainer:
    def test_load_data_triplets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "triplets.jsonl"
            _make_triplet_jsonl(data_path, count=10)

            config = FinetuneConfig()
            trainer = EmbeddingTrainer(config, "test/model")
            train_ds, eval_ds = trainer.load_data(data_path)

            # 10 条数据 → 8 train, 2 eval
            assert len(train_ds) == 8
            assert len(eval_ds) == 2
            assert "anchor" in train_ds.column_names
            assert "positive" in train_ds.column_names

    def test_load_data_small_dataset(self):
        """数据量小于 2 条时代码不崩溃（split 至少保留 1 条 train）"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "small.jsonl"
            _make_triplet_jsonl(data_path, count=3)

            config = FinetuneConfig()
            trainer = EmbeddingTrainer(config, "test/model")
            train_ds, eval_ds = trainer.load_data(data_path)

            assert len(train_ds) >= 1

    def test_output_name_in_run(self):
        """验证 run() 使用正确的 output_name"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "triplets.jsonl"
            _make_triplet_jsonl(data_path, count=5)

            config = FinetuneConfig(output_dir=tmp / "finetuned")
            trainer = EmbeddingTrainer(config, "test/model")

            trainer._output_name = "test-embed"
            out = trainer._get_output_dir()
            assert out == tmp / "finetuned" / "test-embed"
