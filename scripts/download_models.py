"""下载 RAG 所需模型（Embedding + Reranker + LLM）

用法:
    python scripts/download_models.py                  # 下载全部
    python scripts/download_models.py --type embedding  # 仅下载 embedding
    python scripts/download_models.py --type rerank     # 仅下载 reranker
    python scripts/download_models.py --status          # 查看下载状态
"""

import argparse
import os
import sys

from _common import PROJECT_ROOT, banner, fail, info, ok, warn

sys.path.insert(0, str(PROJECT_ROOT))


def run(model_type: str = "all") -> bool:
    """下载模型。

    Args:
        model_type: "all" | "embedding" | "rerank" | "llm"

    Returns:
        True 表示全部下载成功
    """
    # 确保配置已初始化（必须在 import model 之前设置）
    os.environ.setdefault("ENV", "dev")

    from model import models

    # 显示当前状态
    status = models.status()
    info("当前模型状态:")
    for mt, downloaded in status.items():
        mark = "✓" if downloaded else "✗"
        model_id = models.get_default_model_id(mt) or "N/A"
        info(f"  {mark} {mt}: {model_id}")

    if model_type == "all":
        types_to_download = list(status.keys())
    else:
        if model_type not in status:
            fail(f"不支持的模型类型: {model_type}，可选: {list(status.keys())}")
            return False
        types_to_download = [model_type]

    # 过滤已下载的
    pending = [t for t in types_to_download if not status.get(t, False)]
    if not pending:
        ok("所有模型已下载")
        return True

    info(f"\n待下载: {', '.join(pending)}")
    all_ok = True

    for mt in pending:
        info(f"\n正在下载 {mt}...")
        try:
            models.download(mt)
            ok(f"{mt} 下载完成")
        except Exception as e:
            fail(f"{mt} 下载失败: {e}")
            all_ok = False

    if all_ok:
        ok("全部模型下载完成")
    else:
        warn("部分模型下载失败，可稍后重试: python scripts/download_models.py")

    return all_ok


def show_status() -> None:
    """显示模型下载状态"""
    # 确保配置已初始化（必须在 import model 之前设置）
    os.environ.setdefault("ENV", "dev")

    from model import models

    status = models.status()
    info("模型下载状态:")
    for mt, downloaded in status.items():
        mark = "✓" if downloaded else "✗"
        model_id = models.get_default_model_id(mt) or "N/A"
        info(f"  {mark} {mt}: {model_id}")



def main() -> None:
    parser = argparse.ArgumentParser(description="下载 RAG 模型")
    parser.add_argument("--type", default="all", choices=["all", "embedding", "rerank", "llm"],
                        help="模型类型 (默认: all)")
    parser.add_argument("--status", action="store_true", help="仅查看下载状态")
    args = parser.parse_args()

    banner("下载模型")

    if args.status:
        show_status()
        return

    success = run(model_type=args.type)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
