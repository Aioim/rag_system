"""LLMTrainer 单元测试"""

import tempfile
import json
from pathlib import Path

import pytest

from model.finetune.config import FinetuneConfig
from model.finetune.llm_trainer import LLMTrainer
from model.finetune.data import DataValidationError


def _make_llm_jsonl(path: Path, count: int = 10) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({
                "instruction": f"根据文档回答问题{i}",
                "input": f"文档内容第{i}段",
                "output": f"答案{i}",
            }, ensure_ascii=False) + "\n")


def _make_llm_with_teacher_jsonl(path: Path, count: int = 10) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({
                "instruction": f"根据文档回答问题{i}",
                "input": f"文档内容第{i}段",
                "output": f"答案{i}",
                "teacher_output": f"教师答案{i}",
            }, ensure_ascii=False) + "\n")


class TestLLMTrainer:
    def test_sft_load_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "instructions.jsonl"
            _make_llm_jsonl(data_path, count=10)

            config = FinetuneConfig()
            # teacher_model=None → 纯 SFT 模式
            trainer = LLMTrainer(config, "test/model", teacher_model=None)
            train_ds, eval_ds = trainer.load_data(data_path)

            assert len(train_ds) == 8
            assert len(eval_ds) == 2
            assert "instruction" in train_ds.column_names
            assert "output" in train_ds.column_names

    def test_distill_load_data_with_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "instructions_with_teacher.jsonl"
            _make_llm_with_teacher_jsonl(data_path, count=10)

            config = FinetuneConfig()
            trainer = LLMTrainer(config, "test/model", teacher_model="claude-sonnet-5")
            train_ds, eval_ds = trainer.load_data(data_path)

            assert len(train_ds) == 8
            assert "teacher_output" in train_ds.column_names

    def test_distill_missing_teacher_output_raises(self):
        """蒸馏模式下数据缺少 teacher_output 字段应报错"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "instructions.jsonl"
            _make_llm_jsonl(data_path, count=5)

            config = FinetuneConfig()
            trainer = LLMTrainer(config, "test/model", teacher_model="claude-sonnet-5")
            with pytest.raises(DataValidationError, match="teacher_output"):
                trainer.load_data(data_path)

    def test_generate_teacher_requires_teacher_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "instructions.jsonl"
            _make_llm_jsonl(data_path, count=3)

            config = FinetuneConfig()
            trainer = LLMTrainer(config, "test/model", teacher_model=None)
            with pytest.raises(ValueError, match="teacher_model"):
                trainer.generate_teacher_labels(data_path)

    def test_format_prompt(self):
        config = FinetuneConfig()
        trainer = LLMTrainer(config, "test/model")
        prompt = trainer._format_prompt("翻译", "hello", "你好")
        assert "### 指令:" in prompt
        assert "hello" in prompt
        assert "你好" in prompt

    def test_format_prompt_empty_input(self):
        config = FinetuneConfig()
        trainer = LLMTrainer(config, "test/model")
        prompt = trainer._format_prompt("说一下你是谁", "", "我是AI助手")
        assert "### 输入:" not in prompt
        assert "### 指令:" in prompt
        assert "我是AI助手" in prompt

    def test_output_name_in_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "instructions.jsonl"
            _make_llm_jsonl(data_path, count=5)

            config = FinetuneConfig(output_dir=tmp / "finetuned")
            trainer = LLMTrainer(config, "test/model")

            trainer._output_name = "test-llm"
            out = trainer._get_output_dir()
            assert out == tmp / "finetuned" / "test-llm"
