"""数据加载 & 验证测试"""

import json
import tempfile
from pathlib import Path

import pytest

from model.finetune.data import (
    DataValidationError,
    load_jsonl,
    validate_embedding_data,
    validate_reranker_data,
    validate_llm_data,
    split_train_eval,
)


def _write_jsonl(records: list[dict]) -> Path:
    """辅助：写临时 JSONL 文件"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in records:
        tmp.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.close()
    return Path(tmp.name)


class TestLoadJsonl:
    def test_load_valid_file(self):
        records = [{"a": 1}, {"b": 2}]
        path = _write_jsonl(records)
        result = load_jsonl(path)
        assert result == records

    def test_load_empty_file_raises(self):
        path = _write_jsonl([])
        with pytest.raises(DataValidationError, match="为空"):
            load_jsonl(path)

    def test_load_skips_blank_lines(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.write('{"a": 1}\n\n{"b": 2}\n')
        tmp.close()
        result = load_jsonl(Path(tmp.name))
        assert len(result) == 2

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl(Path("/nonexistent/path.jsonl"))

    def test_load_non_jsonl_suffix_raises(self):
        path = _write_jsonl([{"a": 1}])
        # 重命名为 .json
        new_path = path.with_suffix(".json")
        path.rename(new_path)
        with pytest.raises(DataValidationError, match=".jsonl"):
            load_jsonl(new_path)

    def test_load_invalid_json_raises(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.write('not valid json\n')
        tmp.close()
        with pytest.raises(DataValidationError, match="JSON"):
            load_jsonl(Path(tmp.name))


class TestValidateEmbedding:
    def test_valid_triplets(self):
        validate_embedding_data([
            {"query": "q1", "positive": "p1", "negative": "n1"},
            {"query": "q2", "positive": "p2", "negative": "n2"},
        ])

    def test_missing_field_raises(self):
        with pytest.raises(DataValidationError, match="positive"):
            validate_embedding_data([{"query": "q1", "negative": "n1"}])

    def test_empty_field_raises(self):
        with pytest.raises(DataValidationError, match="'query'"):
            validate_embedding_data([{"query": "  ", "positive": "p", "negative": "n"}])


class TestValidateReranker:
    def test_valid_records(self):
        validate_reranker_data([
            {"query": "q1", "document": "d1", "label": 1},
            {"query": "q2", "document": "d2", "label": 0},
        ])

    def test_invalid_label_raises(self):
        with pytest.raises(DataValidationError, match="label"):
            validate_reranker_data([{"query": "q", "document": "d", "label": 2}])


class TestValidateLlm:
    def test_valid_records(self):
        validate_llm_data([
            {"instruction": "instr", "input": "input text", "output": "answer"},
        ])

    def test_empty_output_raises(self):
        with pytest.raises(DataValidationError, match="output"):
            validate_llm_data([{"instruction": "i", "input": "", "output": ""}])

    def test_missing_instruction_raises(self):
        with pytest.raises(DataValidationError, match="instruction"):
            validate_llm_data([{"input": "x", "output": "y"}])


class TestSplitTrainEval:
    def test_default_80_20_split(self):
        records = [{"id": i} for i in range(100)]
        train, ev = split_train_eval(records)
        assert len(train) == 80
        assert len(ev) == 20

    def test_zero_eval_ratio(self):
        records = [{"id": i} for i in range(10)]
        train, ev = split_train_eval(records, eval_ratio=0.0)
        assert len(train) == 10
        assert len(ev) == 0

    def test_small_dataset_min_one_train(self):
        records = [{"id": 1}, {"id": 2}]
        train, ev = split_train_eval(records, eval_ratio=0.5)
        assert len(train) == 1
        assert len(ev) == 1

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            split_train_eval([], eval_ratio=1.5)
