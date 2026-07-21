"""
demo_model.py — 模型下载管理模块演示

演示内容：
  1. 模型管理器单例 (models)
  2. 下载状态查询
  3. 已下载模型列表
  4. 模型路径查询
  5. 微调模型管理（列表/查询/删除）
  6. 下载策略与镜像配置

运行方式：
  cd rag0709
  python examples/06_model/demo_model.py

注意：
  - 本演示为只读模式，不会触发实际下载
  - 如需下载模型，请在 Python REPL 中调用 models.download("embedding")
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402, F401
_ = settings.env  # 强制触发 initialize()，在导入 model 前完成 _config 设置  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    from model import models

    # ── 1. 模型管理器单例 ───────────────────────────────────────
    banner("1. 模型管理器单例")

    print(f"  models 实例: {models!r}")
    print(f"  类型: {type(models).__name__}")
    print(f"  模块版本: 1.1.0")

    # ── 2. 下载状态查询 ─────────────────────────────────────────
    banner("2. 下载状态查询 (models.status)")

    status = models.status()
    print("  各模型下载状态:")
    for model_type, is_downloaded in status.items():
        icon = "✅" if is_downloaded else "⬜"
        print(f"    {icon} {model_type}: {'已下载' if is_downloaded else '未下载'}")

    # ── 3. 已下载模型列表 ───────────────────────────────────────
    banner("3. 已下载模型列表 (models.list_downloaded)")

    downloaded = models.list_downloaded()
    if downloaded:
        print(f"  已下载 {len(downloaded)} 个模型:")
        for model_id, local_path in downloaded.items():
            print(f"    {model_id}")
            print(f"      → {local_path}")
    else:
        print("  暂无已下载的模型")
        print()
        print("  下载模型的几种方式:")
        print("    1. Python: models.download('embedding')")
        print("    2. Python: models.download_all()")
        print("    3. 配置 model.download_source=auto 自动选择 HF/ModelScope")

    # ── 4. 模型路径查询 ─────────────────────────────────────────
    banner("4. 模型路径查询 (models.get_path)")

    for model_type in ["embedding", "rerank", "llm"]:
        path = models.get_path(model_type)
        if path:
            print(f"  {model_type}: {path}")
        else:
            print(f"  {model_type}: 未下载 (返回 None, 不触发下载)")

    print()
    print("  说明: get_path() 是纯查询，不会触发下载。")
    print("  各模型的预期路径结构: local_models/{org}/{model_name}/")

    # ── 5. 配置信息 ─────────────────────────────────────────────
    banner("5. 模型下载配置")

    from config import settings

    print(f"  缓存目录:      {settings.model.cache_dir}")
    print(f"  下载源:        {settings.model.download_source}")
    print(f"  HF 端点:       {settings.model.hf_endpoint}")
    print(f"  最大重试:      {settings.model.max_retries} 次")
    print(f"  Token 环境变量: {settings.model.hf_token_env}")
    print()
    print(f"  默认模型:")
    print(f"    embedding:    {settings.model.default_models['embedding']}")
    print(f"    rerank:       {settings.model.default_models['rerank']}")
    print(f"    llm (本地):   {settings.model.default_models['llm']}")

    # ── 6. 微调模型管理 ─────────────────────────────────────────
    banner("6. 微调模型管理 (models.list_finetuned)")

    finetuned = models.list_finetuned()
    if finetuned:
        print(f"  已微调 {len(finetuned)} 个适配器:")
        for name, info in finetuned.items():
            print(f"    {name}: {info}")
    else:
        print("  暂无已微调的适配器")
        print()
        print("  微调模型的方式:")
        print("    # Embedding 对比学习微调")
        print("    models.finetune('embedding', data_path='data/finetune/triplets.jsonl')")
        print()
        print("    # Reranker 分类微调")
        print("    models.finetune('reranker', data_path='data/finetune/rerank_data.jsonl')")
        print()
        print("    # LLM SFT + 蒸馏")
        print("    models.finetune('llm', data_path='data/finetune/instructions.jsonl',")
        print("                    teacher='deepseek-v4-pro', alpha=0.3)")

    # ── 7. 微调配置预览 ─────────────────────────────────────────
    banner("7. 微调配置预览")

    ft = settings.finetune
    print(f"  输出目录:        {ft.output_dir}")
    print(f"  设备:            {ft.device}")
    print(f"  训练轮次:        {ft.training.epochs}")
    print(f"  学习率:          {ft.training.learning_rate}")
    print(f"  批次大小:        {ft.training.batch_size}")
    print(f"  LoRA rank:       {ft.lora.r}")
    print(f"  LoRA alpha:       {ft.lora.lora_alpha}")
    print(f"  蒸馏温度:        {ft.distillation.temperature}")
    print(f"  蒸馏 alpha:      {ft.distillation.alpha}")
    print()
    print(f"  查看完整配置: python -m model.finetune config")

    # ── 8. 下载策略 ─────────────────────────────────────────────
    banner("8. 下载策略")

    from model.downloader import AutoStrategy, HfStrategy, MsStrategy

    print("  可用策略:")
    print(f"    AutoStrategy — 自动选择 HF/ModelScope")
    print(f"    HfStrategy   — HuggingFace Hub（国内可用 hf-mirror.com 镜像）")
    print(f"    MsStrategy   — ModelScope（国内速度更优）")
    print()
    print(f"  当前策略: {settings.model.download_source}")
    print(f"  切换: 修改 config/dev.yaml → model.download_source = 'modelscope'")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 模型管理模块演示完成")
    print()
    print("  CLI 工具:")
    print("    python -m model.finetune <type> --data <path> [--name <n>] [--teacher <t>]")
    print("    python -m model.finetune config  # 查看微调配置")
    print()
    print("  训练数据格式 (JSONL):")
    print("    embedding: query, positive, negative")
    print("    reranker:  query, document, label (0/1)")
    print("    llm:       instruction, input, output")


if __name__ == "__main__":
    main()
