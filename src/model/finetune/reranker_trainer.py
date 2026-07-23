"""
Reranker 模型微调 — CrossEncoder + LoRA 二分类微调
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoTokenizer as AutoTokenizerType

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .base import BaseTrainer
from .config import FinetuneConfig
from .data import load_jsonl


class RerankerTrainer(BaseTrainer):
    """微调 BGE-Reranker CrossEncoder 模型

    基座: BAAI/bge-reranker-v2-m3
    损失: CrossEntropyLoss（相关/不相关二分类）
    """

    model_type = "reranker"

    def __init__(self, config: FinetuneConfig, base_model_id: str):
        super().__init__(config, base_model_id)
        self._tokenizer: AutoTokenizerType | None = None  # 延迟加载

    def load_data(self, data_path: Path, records: list[dict] | None = None) -> tuple[Dataset, Dataset | None]:
        from .data import split_train_eval

        if records is None:
            records = load_jsonl(data_path)
        # 数据格式校验由 BaseTrainer.run() → _validate_records() 统一执行，此处不再重复

        train_records, eval_records = split_train_eval(records, eval_ratio=0.2)

        def _to_dataset(recs: list[dict]) -> Dataset:
            return Dataset.from_dict({
                "query": [r["query"] for r in recs],
                "document": [r["document"] for r in recs],
                "label": [r["label"] for r in recs],
            })

        train_ds = _to_dataset(train_records)
        eval_ds = _to_dataset(eval_records) if eval_records else None
        return train_ds, eval_ds

    def _tokenize(self, examples: dict) -> dict:
        """将 (query, document) 对 tokenize"""
        tokenized = self._tokenizer(
            examples["query"],
            examples["document"],
            truncation=True,
            padding="max_length",
            max_length=self.config.training.max_seq_length,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    def train(self, train_dataset: Dataset, eval_dataset: Dataset | None = None) -> Path:
        # 1. 加载 tokenizer 和模型
        device = self._resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_id, num_labels=2
        )
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            else:
                self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(self._tokenizer))
        model.config.pad_token_id = self._tokenizer.pad_token_id

        # 2. 注入 LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.to(device)

        # 3. Tokenize 数据集
        train_ds = train_dataset.map(self._tokenize, batched=True, remove_columns=["query", "document", "label"])
        eval_ds = None
        if eval_dataset is not None:
            eval_ds = eval_dataset.map(self._tokenize, batched=True, remove_columns=["query", "document", "label"])

        # 4. 训练参数
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

        # 5. 训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
        trainer.train()

        # 6. 保存 LoRA 适配器
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))

        self._metrics = {"eval_loss": getattr(trainer.state, "best_metric", None)}
        return output_dir
