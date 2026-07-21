"""
LLM 微调 & 蒸馏 — SFT 微调 + 云端大模型黑盒蒸馏
"""

import json
import os
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from logger import logger

if TYPE_CHECKING:
    from transformers import AutoTokenizer as AutoTokenizerType

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .base import BaseTrainer
from .config import FinetuneConfig
from .data import DataValidationError, load_jsonl, split_train_eval, validate_llm_data


class DistillationTrainer(Trainer):
    """自定义 Trainer，支持硬标签 + 教师标签混合损失"""

    def __init__(self, alpha: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_labels = inputs.get("teacher_labels")
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for causal LM: (batch, seq, vocab) → (batch, seq-1, vocab)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 与 transformers 约定一致：提供 num_items_in_batch 时用 sum/num_items
        # 归一化（梯度累积下正确），否则退回 token 平均（mean）
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=-100)
        hard_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        if num_items_in_batch is not None:
            hard_loss = hard_loss / num_items_in_batch

        if teacher_labels is not None:
            shift_teacher = teacher_labels[..., 1:].contiguous()
            distill_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_teacher.view(-1),
            )
            if num_items_in_batch is not None:
                distill_loss = distill_loss / num_items_in_batch

            loss = self.alpha * hard_loss + (1.0 - self.alpha) * distill_loss
        else:
            loss = hard_loss

        return (loss, outputs) if return_outputs else loss


class LLMTrainer(BaseTrainer):
    """LLM 微调/蒸馏

    基座: Qwen3-0.6B (默认)
    教师: 云端 API（DeepSeek/GPT 等）

    蒸馏流程:
      1. generate_teacher_labels(data_path) → 用教师模型生成答案
      2. train() → 混合硬标签 + 教师标签训练
    """

    model_type = "llm"
    _TEACHER_OUTPUT_FIELD = "teacher_output"
    # 相邻教师 API 调用之间的固定间隔（秒），降低云端速率限制（429）触发概率
    _TEACHER_CALL_INTERVAL_SECONDS = 0.5

    def __init__(self, config: FinetuneConfig, base_model_id: str,
                 teacher_model: str | None = None):
        super().__init__(config, base_model_id)
        self.teacher_model = teacher_model  # None=纯SFT
        self._tokenizer: AutoTokenizerType | None = None

    # ---- 蒸馏第1步：教师标签生成 ----

    def generate_teacher_labels(self, data_path: Path) -> Path:
        """用云端教师模型为每条数据生成答案。

        输入: instructions.jsonl
        输出: 同目录下的 instructions_with_teacher.jsonl

        断点续传: 检查 teacher_output 字段是否已存在，
        已存在的跳过，便于 API 调用失败后重试。进度逐行落盘到 .tmp，
        中断后下次运行可从 .tmp 恢复已生成的标签。
        """
        if self.teacher_model is None:
            raise ValueError("必须指定 teacher_model 参数才能生成教师标签")

        records = load_jsonl(data_path)
        validate_llm_data(records)

        output_path = data_path.parent / f"{data_path.stem}_with_teacher.jsonl"
        tmp_path = output_path.with_suffix(".jsonl.tmp")

        # 断点续传：加载已有进度（含上次中断遗留的 .tmp 部分进度）
        existing = self._load_teacher_progress([output_path, tmp_path])

        made_call = False
        with open(tmp_path, "w", encoding="utf-8") as f:
            for i, r in enumerate(records):
                key = (r.get("instruction", ""), r.get("input", ""))
                if key in existing:
                    r[self._TEACHER_OUTPUT_FIELD] = existing[key]
                else:
                    # 简单限流：相邻调用之间固定间隔
                    if made_call and self._TEACHER_CALL_INTERVAL_SECONDS > 0:
                        time.sleep(self._TEACHER_CALL_INTERVAL_SECONDS)
                    teacher_answer = self._call_teacher(r["instruction"], r["input"])
                    r[self._TEACHER_OUTPUT_FIELD] = teacher_answer
                    made_call = True

                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()  # 逐行落盘：中断时已生成的标签可从 .tmp 恢复

                # 进度日志
                if (i + 1) % 10 == 0:
                    logger.info(f"教师标签生成进度: {i + 1}/{len(records)}")

        os.replace(tmp_path, output_path)
        return output_path

    def _load_teacher_progress(self, paths: list[Path]) -> dict:
        """读取既有教师标签进度；后出现的文件覆盖先出现的（.tmp 更新）。

        容忍中断产生的残缺行（跳过无法解析的 JSON）。
        """
        existing: dict = {}
        for path in paths:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as pf:
                for line in pf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 中断残留的半行
                    if self._TEACHER_OUTPUT_FIELD in r:
                        key = (r.get("instruction", ""), r.get("input", ""))
                        existing[key] = r[self._TEACHER_OUTPUT_FIELD]
        return existing

    def _call_teacher(self, instruction: str, input_text: str) -> str:
        """调用云端教师模型生成答案。

        优先走项目的 LLM 路由模块；若模块未就绪，
        则使用 DEEPSEEK_API_KEY 环境变量直接调 DeepSeek API 作为过渡。
        """
        # 尝试走项目 LLM 路由
        try:
            from langchain.schema import HumanMessage

            from generation.llm_router import route_llm
            response = route_llm(
                model=self.teacher_model,
                messages=[HumanMessage(content=f"{instruction}\n\n{input_text}")],
            )
            return response.content
        except (ImportError, AttributeError):
            pass  # LLM 路由模块未就绪，降级到 DeepSeek API 直调
        except Exception as e:
            logger.warning("LLM 路由调用失败: %s，降级到 DeepSeek API 直调", e)

        # 过渡方案：直接调 DeepSeek API
        # 优先通过 secrets_manager 获取，降级到环境变量
        try:
            from security import secrets as _sec
            secret_obj = _sec.get_secret("DEEPSEEK_API_KEY", required=False)
            api_key = str(secret_obj) if secret_obj else os.getenv("DEEPSEEK_API_KEY")
        except Exception:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "教师模型 API 密钥未配置。请设置 DEEPSEEK_API_KEY 环境变量，"
                "或等待 src/generation/ 模块实现后走 LLM 路由。"
            )

        from openai import OpenAI
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=60.0)
                message = client.chat.completions.create(
                    model=self.teacher_model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": f"{instruction}\n\n{input_text}"}],
                )
                return message.choices[0].message.content
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    # ---- 数据加载 ----

    def load_data(self, data_path):
        records = load_jsonl(data_path)
        validate_llm_data(records)

        # 蒸馏模式需要 teacher_output 字段
        if self.teacher_model is not None:
            for i, r in enumerate(records, start=1):
                if self._TEACHER_OUTPUT_FIELD not in r:
                    raise DataValidationError(
                        f"蒸馏模式下需要 '{self._TEACHER_OUTPUT_FIELD}' 字段，"
                        f"请先运行 generate_teacher_labels()",
                        i,
                    )

        train_records, eval_records = split_train_eval(records, eval_ratio=0.2)

        # 警告：数据含 teacher_output 但 teacher_model 未设置
        if self.teacher_model is None:
            has_teacher_field = any(self._TEACHER_OUTPUT_FIELD in r for r in records)
            if has_teacher_field:
                warnings.warn(
                    f"数据中包含 '{self._TEACHER_OUTPUT_FIELD}' 字段，但 teacher_model 未设置。"
                    f"将执行纯 SFT 训练，教师标签将被忽略。如需蒸馏请传入 teacher 参数。",
                    stacklevel=2,
                )

        def _to_dataset(recs: list[dict]) -> Dataset:
            data = {
                "instruction": [r["instruction"] for r in recs],
                "input": [r["input"] for r in recs],
                "output": [r["output"] for r in recs],
            }
            if self.teacher_model is not None:
                data["teacher_output"] = [r.get(self._TEACHER_OUTPUT_FIELD, r["output"]) for r in recs]
            return Dataset.from_dict(data)

        train_ds = _to_dataset(train_records)
        eval_ds = _to_dataset(eval_records) if eval_records else None
        return train_ds, eval_ds

    def _format_prompt(self, instruction: str, input_text: str, output_text: str = "") -> str:
        """格式化为训练 prompt 模板"""
        if input_text:
            text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output_text}"
        else:
            text = f"### 指令:\n{instruction}\n\n### 回答:\n{output_text}"
        return text

    def _tokenize(self, examples: dict) -> dict:
        """Tokenize for CausalLM training, supports teacher labels for distillation"""
        # Human labels
        human_prompts = [
            self._format_prompt(inst, inp, out)
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"], strict=True
            )
        ]
        tokenized = self._tokenizer(
            human_prompts,
            truncation=True,
            padding="max_length",
            max_length=self.config.training.max_seq_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()

        # Mask input tokens — only response tokens should contribute to loss
        prefix_texts = [
            self._format_prompt(inst, inp, "")
            for inst, inp in zip(examples["instruction"], examples["input"], strict=True)
        ]
        prefix_tokenized = self._tokenizer(
            prefix_texts,
            truncation=True,
            max_length=self.config.training.max_seq_length,
        )
        for i in range(len(tokenized["labels"])):
            prefix_len = len(prefix_tokenized["input_ids"][i])
            tokenized["labels"][i][:prefix_len] = [-100] * prefix_len

        # Teacher labels (distillation mode)
        has_teacher = "teacher_output" in examples and any(t for t in examples["teacher_output"] if t)
        if has_teacher:
            teacher_prompts = [
                self._format_prompt(inst, inp, t_out)
                for inst, inp, t_out in zip(
                    examples["instruction"], examples["input"], examples["teacher_output"], strict=True
                )
            ]
            teacher_tokenized = self._tokenizer(
                teacher_prompts,
                truncation=True,
                padding="max_length",
                max_length=self.config.training.max_seq_length,
            )
            tokenized["teacher_labels"] = teacher_tokenized["input_ids"].copy()

            # Mask input tokens for teacher labels too
            for i in range(len(tokenized["teacher_labels"])):
                prefix_len = len(prefix_tokenized["input_ids"][i])
                tokenized["teacher_labels"][i][:prefix_len] = [-100] * prefix_len

        return tokenized

    # ---- 训练 ----

    def train(self, train_dataset: Dataset, eval_dataset: Dataset | None = None) -> Path:
        device = self._resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        # 注入 LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.to(device)

        # Tokenize
        is_distill = "teacher_output" in train_dataset.column_names
        remove_cols = ["instruction", "input", "output"]
        if is_distill:
            remove_cols.append("teacher_output")

        train_ds = train_dataset.map(
            self._tokenize, batched=True,
            remove_columns=remove_cols,
        )
        eval_ds = None
        if eval_dataset is not None:
            eval_ds = eval_dataset.map(
                self._tokenize, batched=True,
                remove_columns=remove_cols,
            )

        output_dir = self._get_output_dir()
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            warmup_ratio=self.config.training.warmup_ratio,
            eval_strategy="steps" if eval_ds else "no",
            eval_steps=self.config.training.eval_steps,
            save_strategy="steps",
            save_steps=self.config.training.save_steps,
            logging_steps=self.config.training.logging_steps,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            save_total_limit=2,
            load_best_model_at_end=eval_ds is not None,
            report_to="none",
            remove_unused_columns=False,
        )

        if is_distill:
            trainer = DistillationTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer=self._tokenizer, model=model, padding=True
                ),
                alpha=self.config.distillation.alpha,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer=self._tokenizer, model=model, padding=True
                ),
            )
        trainer.train()

        # 保存
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))

        self._metrics = {"eval_loss": getattr(trainer.state, "best_metric", None)}
        return output_dir
