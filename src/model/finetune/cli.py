"""
微调 CLI 入口 — python -m model.finetune <subcommand> [args]
"""

import argparse
import sys
from pathlib import Path

from .config import get_finetune_config, FinetuneConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m model.finetune",
        description="RAG 模型微调 & 蒸馏工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- embedding ---
    emb = subparsers.add_parser("embedding", help="微调 Embedding 模型")
    _add_common_args(emb)
    emb.add_argument("--base-model", default=None,
                     help="基座模型 repo_id（默认从配置读取）")

    # --- reranker ---
    rnk = subparsers.add_parser("reranker", help="微调 Reranker 模型")
    _add_common_args(rnk)
    rnk.add_argument("--base-model", default=None,
                     help="基座模型 repo_id（默认从配置读取）")

    # --- llm ---
    llm = subparsers.add_parser("llm", help="微调/蒸馏 LLM")
    _add_common_args(llm)
    llm.add_argument("--base-model", default=None,
                     help="基座模型 repo_id（默认 Qwen3-0.6B）")
    llm.add_argument("--teacher", default=None,
                     help="教师模型 ID（云端 API），指定后启用蒸馏模式")
    llm.add_argument("--alpha", type=float, default=None,
                     help="硬标签权重（0=纯蒸馏，1=纯SFT，默认 0.5）")
    llm.add_argument("--generate-only", action="store_true",
                     help="只生成教师标签，不执行训练")

    # --- list ---
    subparsers.add_parser("list", help="列出所有已微调适配器")

    # --- remove ---
    rem = subparsers.add_parser("remove", help="删除指定适配器")
    rem.add_argument("--name", required=True, help="适配器名称")

    # --- info ---
    info = subparsers.add_parser("info", help="查看适配器详情")
    info.add_argument("--name", required=True, help="适配器名称")

    return parser


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """所有训练子命令的公共参数"""
    p.add_argument("--data", type=Path, required=True, help="训练数据 JSONL 路径")
    p.add_argument("--name", default=None, help="适配器输出名称（默认自动生成）")
    p.add_argument("--epochs", type=int, default=None, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=None, help="每批大小")
    p.add_argument("--lr", type=float, default=None, help="学习率")
    p.add_argument("--output-dir", type=Path, default=None, help="适配器输出目录")
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])


def _apply_overrides(config: FinetuneConfig, args: argparse.Namespace) -> None:
    """CLI 参数覆盖配置（CLI > 环境变量 > YAML）"""
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if hasattr(args, "alpha") and args.alpha is not None:
        config.distillation.alpha = args.alpha


from .aliases import _MODEL_TYPE_ALIASES


def _get_base_model(model_type: str, args) -> str:
    """解析基座模型 ID"""
    base = getattr(args, "base_model", None)
    if base:
        return base
    resolved = _MODEL_TYPE_ALIASES.get(model_type, model_type)
    try:
        from model import models
        models._ensure_init()
        return models._defaults.get(resolved, "")
    except (ImportError, AttributeError, KeyError) as e:
        import sys
        print(f"警告: 无法从配置获取默认模型: {e}", file=sys.stderr)
    return ""


def main(argv: list[str] | None = None) -> None:
    """CLI 主入口"""
    parser = _build_parser()
    args = parser.parse_args(argv or sys.argv[1:])

    if args.command == "list":
        _cmd_list()
    elif args.command == "remove":
        _cmd_remove(args.name)
    elif args.command == "info":
        _cmd_info(args.name)
    elif args.command in ("embedding", "reranker", "llm"):
        _cmd_train(args)
    else:
        parser.print_help()


def _cmd_train(args) -> None:
    """执行训练命令"""
    from .embedding_trainer import EmbeddingTrainer
    from .reranker_trainer import RerankerTrainer
    from .llm_trainer import LLMTrainer

    config = get_finetune_config()
    _apply_overrides(config, args)

    base_model = _get_base_model(args.command, args)
    if not base_model:
        print(f"错误: 无法确定 {args.command} 的基座模型，请用 --base-model 指定")
        sys.exit(1)

    trainer_classes = {
        "embedding": EmbeddingTrainer,
        "reranker": RerankerTrainer,
        "llm": LLMTrainer,
    }
    trainer_cls = trainer_classes[args.command]

    # LLM 蒸馏模式
    if args.command == "llm" and args.teacher:
        trainer = LLMTrainer(config, base_model, teacher_model=args.teacher)

        if args.generate_only:
            output = trainer.generate_teacher_labels(args.data)
            print(f"教师标签已生成: {output}")
            return

        # 检查数据是否已有教师标签
        output = trainer.run(args.data, output_name=args.name)
    else:
        trainer = trainer_cls(config, base_model)
        output = trainer.run(args.data, output_name=args.name)

    print(f"训练完成!")
    print(f"  模型类型: {output.model_type}")
    print(f"  基座模型: {output.base_model}")
    print(f"  适配器路径: {output.adapter_path}")
    print(f"  耗时: {output.duration_seconds:.1f}s")
    if output.metrics:
        print(f"  指标: {output.metrics}")


def _cmd_list() -> None:
    """列出所有已微调适配器"""
    from config.path import PROJECT_ROOT
    from .config import get_finetune_config
    from .base import BaseTrainer

    config = get_finetune_config()
    output_dir = config.resolve_output_dir(PROJECT_ROOT)
    adapters = BaseTrainer.scan_finetuned(output_dir)

    if not adapters:
        print("暂无已微调的适配器")
        return

    print(f"{'名称':<30} {'类型':<12} {'基座模型':<35} {'创建时间'}")
    print("-" * 100)
    for name, info in adapters.items():
        print(f"{name:<30} {info.model_type:<12} {info.base_model:<35} {info.created_at}")


def _cmd_remove(name: str) -> None:
    """删除适配器"""
    import shutil
    from config.path import PROJECT_ROOT
    from .config import get_finetune_config
    from .base import BaseTrainer

    config = get_finetune_config()
    output_dir = config.resolve_output_dir(PROJECT_ROOT)
    adapters = BaseTrainer.scan_finetuned(output_dir)

    if name not in adapters:
        print(f"适配器 '{name}' 不存在")
        sys.exit(1)

    shutil.rmtree(adapters[name].adapter_path)
    print(f"已删除适配器: {name}")


def _cmd_info(name: str) -> None:
    """查看适配器详情"""
    from config.path import PROJECT_ROOT
    from .config import get_finetune_config
    from .base import BaseTrainer

    config = get_finetune_config()
    output_dir = config.resolve_output_dir(PROJECT_ROOT)
    adapters = BaseTrainer.scan_finetuned(output_dir)

    if name not in adapters:
        print(f"适配器 '{name}' 不存在")
        sys.exit(1)

    info = adapters[name]
    print(f"名称:       {info.name}")
    print(f"类型:       {info.model_type}")
    print(f"基座模型:   {info.base_model}")
    print(f"路径:       {info.adapter_path}")
    print(f"创建时间:   {info.created_at}")
    print(f"训练指标:   {info.metrics}")
    print(f"训练参数:   {info.training_config}")


if __name__ == "__main__":
    main()
