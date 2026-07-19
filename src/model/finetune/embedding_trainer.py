"""
Embedding 模型微调 — 基于 sentence-transformers + LoRA
"""

from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import (
    SentenceTransformerTrainingArguments,
)

from .base import BaseTrainer
from .config import FinetuneConfig
from .data import load_jsonl, validate_embedding_data


class EmbeddingTrainer(BaseTrainer):
    """微调 BGE Embedding 模型

    基座模型: BAAI/bge-large-zh-v1.5
    损失函数: MultipleNegativesRankingLoss
    """

    model_type = "embedding"

    def __init__(self, config: FinetuneConfig, base_model_id: str):
        super().__init__(config, base_model_id)

    def load_data(self, data_path: Path) -> tuple[Dataset, Dataset | None]:
        from .data import split_train_eval

        records = load_jsonl(data_path)
        validate_embedding_data(records)

        # sentence-transformers MultipleNegativesRankingLoss 需要
        # InputExample(texts=[query, positive, negative])
        # 转换为 Dataset 格式: {"anchor": query, "positive": pos}
        # MultipleNegativesRankingLoss 使用 in-batch negatives，
        # 所以只需要 (anchor, positive) 对
        train_records, eval_records = split_train_eval(records, eval_ratio=0.2)

        train_data = {
            "anchor": [r["query"] for r in train_records],
            "positive": [r["positive"] for r in train_records],
        }
        train_ds = Dataset.from_dict(train_data)

        eval_ds = None
        if eval_records:
            eval_data = {
                "anchor": [r["query"] for r in eval_records],
                "positive": [r["positive"] for r in eval_records],
            }
            eval_ds = Dataset.from_dict(eval_data)

        return train_ds, eval_ds

    def train(self, train_dataset: Dataset, eval_dataset: Dataset | None = None) -> Path:
        # 1. 加载基座模型
        device = self._resolve_device()
        model = SentenceTransformer(self.base_model_id, device=str(device))

        # 2. 注入 LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
        )
        # 注: model[0] 返回 SentenceTransformer 的第一个 module，
        # 等价于 _first_module() 但使用公共 API
        first_module = model[0]
        first_module.auto_model = get_peft_model(first_module.auto_model, lora_cfg)

        # 3. 损失函数
        loss = MultipleNegativesRankingLoss(model)

        # 4. 训练参数
        output_dir = self._get_output_dir()
        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            warmup_ratio=self.config.training.warmup_ratio,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.training.eval_steps,
            save_strategy="steps",
            save_steps=self.config.training.save_steps,
            logging_steps=self.config.training.logging_steps,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            save_total_limit=2,
            load_best_model_at_end=eval_dataset is not None,
            report_to="none",
        )

        # 5. 训练
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )
        trainer.train()

        # 6. 保存 LoRA 适配器
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        self._metrics = {"train_loss": getattr(trainer.state, "best_metric", None)}

        return output_dir
