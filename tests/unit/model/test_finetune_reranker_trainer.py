"""RerankerTrainer 单元测试"""

import tempfile
import json
from pathlib import Path

from model.finetune.config import FinetuneConfig
from model.finetune.reranker_trainer import RerankerTrainer


def _make_reranker_jsonl(path: Path, count: int = 10) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({
                "query": f"查询{i}",
                "document": f"文档{i}",
                "label": i % 2,
            }, ensure_ascii=False) + "\n")


class TestRerankerTrainer:
    def test_load_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "rerank_data.jsonl"
            _make_reranker_jsonl(data_path, count=10)

            config = FinetuneConfig()
            trainer = RerankerTrainer(config, "test/model")
            train_ds, eval_ds = trainer.load_data(data_path)

            assert len(train_ds) == 8
            assert len(eval_ds) == 2
            assert "query" in train_ds.column_names
            assert "label" in train_ds.column_names

    def test_output_name_in_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "rerank_data.jsonl"
            _make_reranker_jsonl(data_path, count=5)

            config = FinetuneConfig(output_dir=tmp / "finetuned")
            trainer = RerankerTrainer(config, "test/model")

            trainer._output_name = "test-rerank"
            out = trainer._get_output_dir()
            assert out == tmp / "finetuned" / "test-rerank"
