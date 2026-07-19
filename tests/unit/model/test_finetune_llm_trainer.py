"""LLMTrainer 单元测试"""

import tempfile
import json
from pathlib import Path

import pytest
import torch

from model.finetune.config import FinetuneConfig
from model.finetune.llm_trainer import DistillationTrainer, LLMTrainer
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


def _make_distill_trainer(alpha: float = 0.5) -> DistillationTrainer:
    """绕过 Trainer.__init__（需要完整模型/训练参数），只测 compute_loss 数学逻辑"""
    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.alpha = alpha
    return trainer


class _FakeModel:
    """返回固定 logits 的假模型"""

    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def __call__(self, **inputs):
        return type("_Output", (), {"logits": self._logits})()


def _make_loss_fixtures():
    """固定随机种子生成 logits/labels，labels 前 2 列掩码模拟 prompt 部分"""
    torch.manual_seed(0)
    batch, seq, vocab = 2, 6, 11
    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[:, :2] = -100
    return logits, labels


def _sum_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """shift 后按 sum 归约的交叉熵（transformers 约定的分子）"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
    )


class TestDistillationTrainerComputeLoss:
    """复现审查发现：CrossEntropyLoss(mean) 后又除 num_items_in_batch 导致二次归一化"""

    def test_sft_loss_normalized_once_with_num_items_in_batch(self):
        """提供 num_items_in_batch 时，loss 应等于 sum_loss / num_items（单次归一化）"""
        logits, labels = _make_loss_fixtures()
        shift_labels = labels[..., 1:]
        num_items = int((shift_labels != -100).sum())
        expected = _sum_ce(logits, labels) / num_items

        trainer = _make_distill_trainer()
        loss = trainer.compute_loss(
            _FakeModel(logits), {"labels": labels}, num_items_in_batch=num_items
        )

        assert torch.isclose(loss, expected, rtol=1e-5), (
            f"loss={loss.item():.6f}，应为单次归一化的 {expected.item():.6f}"
        )

    def test_distill_loss_normalized_once_with_num_items_in_batch(self):
        """蒸馏分支：hard/distill 两项均应为 sum/num_items 后再按 alpha 加权"""
        logits, labels = _make_loss_fixtures()
        # 教师标签：非掩码位置整体偏移 1（保持 -100 掩码位置一致）
        teacher_labels = labels.clone()
        mask = teacher_labels != -100
        teacher_labels[mask] = (teacher_labels[mask] + 1) % 11

        shift_labels = labels[..., 1:]
        num_items = int((shift_labels != -100).sum())
        alpha = 0.3
        expected = (
            alpha * _sum_ce(logits, labels) / num_items
            + (1.0 - alpha) * _sum_ce(logits, teacher_labels) / num_items
        )

        trainer = _make_distill_trainer(alpha=alpha)
        loss = trainer.compute_loss(
            _FakeModel(logits),
            {"labels": labels, "teacher_labels": teacher_labels},
            num_items_in_batch=num_items,
        )

        assert torch.isclose(loss, expected, rtol=1e-5), (
            f"loss={loss.item():.6f}，应为 {expected.item():.6f}"
        )

    def test_loss_falls_back_to_mean_without_num_items_in_batch(self):
        """未提供 num_items_in_batch 时保持 token 平均（回归保护）"""
        logits, labels = _make_loss_fixtures()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        expected = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        trainer = _make_distill_trainer()
        loss = trainer.compute_loss(_FakeModel(logits), {"labels": labels})

        assert torch.isclose(loss, expected, rtol=1e-5)


class TestTeacherGenerationResilience:
    """审查 H10：教师标签生成需限流 + 中断后已生成数据可恢复"""

    def test_resumes_from_interrupted_tmp_progress(self, tmp_path, monkeypatch):
        """上次中断残留的 .tmp 进度（含半行残片）应被复用，不重复调教师"""
        # Arrange
        data_path = tmp_path / "instructions.jsonl"
        _make_llm_jsonl(data_path, count=3)

        rec0 = {
            "instruction": "根据文档回答问题0", "input": "文档内容第0段",
            "output": "答案0", "teacher_output": "已有教师答案0",
        }
        tmp_progress = tmp_path / "instructions_with_teacher.jsonl.tmp"
        tmp_progress.write_text(
            json.dumps(rec0, ensure_ascii=False) + "\n" + '{"instruction": "半行',
            encoding="utf-8",
        )

        trainer = LLMTrainer(
            FinetuneConfig(), "test/model", teacher_model="claude-sonnet-5"
        )
        calls: list[str] = []
        monkeypatch.setattr(
            trainer, "_call_teacher",
            lambda ins, inp: (calls.append(ins), "新教师答案")[1],
        )
        monkeypatch.setattr("time.sleep", lambda s: None)

        # Act
        out = trainer.generate_teacher_labels(data_path)

        # Assert
        assert len(calls) == 2, "第 0 条应复用中断前的进度，不再调教师"
        results = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
        ]
        assert results[0]["teacher_output"] == "已有教师答案0"
        assert all("teacher_output" in r for r in results)

    def test_throttles_between_teacher_calls(self, tmp_path, monkeypatch):
        """相邻教师 API 调用之间应有限流间隔"""
        # Arrange
        data_path = tmp_path / "instructions.jsonl"
        _make_llm_jsonl(data_path, count=3)

        trainer = LLMTrainer(
            FinetuneConfig(), "test/model", teacher_model="claude-sonnet-5"
        )
        monkeypatch.setattr(trainer, "_call_teacher", lambda ins, inp: "答")
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

        # Act
        trainer.generate_teacher_labels(data_path)

        # Assert：3 次全新调用 → 2 次间隔
        assert len(sleeps) == 2, "相邻教师调用之间应有限流 sleep"
        assert all(s > 0 for s in sleeps)
