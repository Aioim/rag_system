"""
02_finetune.py — 模型管理：微调模型管理

演示内容：
  1. 微调模型管理（list_finetuned / get_finetuned_path / remove_finetuned）
  2. 微调配置预览（LoRA / 训练 / 蒸馏）
  3. 训练数据格式说明
  4. CLI 入口与配置查看

运行方式：
  cd rag0709
  python examples/06_model/02_finetune.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402
_ = settings.env  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    from model import models

    # ── 1. 微调模型管理 ─────────────────────────────────────────
    banner("1. 微调模型管理 (models.list_finetuned)")

    finetuned = models.list_finetuned()
    if finetuned:
        print(f"  已微调 {len(finetuned)} 个适配器:")
        for name, info in finetuned.items():
            print(f"    {name}: {info}")
    else:
        print("  暂无已微调的适配器")
        print()
        print("  微调模型的方式:")
        print()
        print("    # Embedding 对比学习微调")
        print("    models.finetune('embedding', data_path='data/finetune/triplets.jsonl')")
        print()
        print("    # Reranker 分类微调")
        print("    models.finetune('reranker', data_path='data/finetune/rerank_data.jsonl')")
        print()
        print("    # LLM SFT + 蒸馏")
        print("    models.finetune('llm', data_path='data/finetune/instructions.jsonl',")
        print("                    teacher='deepseek-v4-pro', alpha=0.3)")

    # ── 2. 微调配置预览 ─────────────────────────────────────────
    banner("2. 微调配置预览")

    ft = settings.finetune
    print(f"  输出目录:        {ft.output_dir}")
    print(f"  设备:            {ft.device}")
    print(f"  训练轮次:        {ft.training.epochs}")
    print(f"  学习率:          {ft.training.learning_rate}")
    print(f"  批次大小:        {ft.training.batch_size}")
    print(f"  LoRA rank:       {ft.lora.r}")
    print(f"  LoRA alpha:      {ft.lora.lora_alpha}")
    print(f"  蒸馏温度:        {ft.distillation.temperature}")
    print(f"  蒸馏 alpha:      {ft.distillation.alpha}")
    print()
    print(f"  查看完整配置: python -m model.finetune config")

    # ── 3. 训练数据格式 ─────────────────────────────────────────
    banner("3. 训练数据格式 (JSONL)")

    print("  每种微调类型需要的 JSONL 字段:")
    print()
    print("  embedding (对比学习):")
    print('    {"query": "...", "positive": "...", "negative": "..."}')
    print()
    print("  reranker (分类):")
    print('    {"query": "...", "document": "...", "label": 1}')
    print()
    print("  llm (SFT + 蒸馏):")
    print('    {"instruction": "...", "input": "...", "output": "..."}')

    # ── 4. CLI 入口 ─────────────────────────────────────────────
    banner("4. CLI 入口")

    print("  # 微调命令")
    print("  python -m model.finetune <type> --data <path> [--name <n>] [--teacher <t>]")
    print()
    print("  # 查看配置")
    print("  python -m model.finetune config")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 微调管理演示完成")


if __name__ == "__main__":
    main()
