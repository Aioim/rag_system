"""
01_status_download.py — 模型管理：下载状态与配置

演示内容：
  1. 模型管理器单例 (models)
  2. 下载状态查询 (status)
  3. 已下载模型列表 (list_downloaded)
  4. 模型路径查询 (get_path，不触发下载)
  5. 下载配置与策略（HF/ModelScope/Auto）

运行方式：
  cd rag0709
  python examples/06_model/01_status_download.py

注意：本演示为只读模式，不会触发实际下载
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

    # ── 1. 模型管理器单例 ───────────────────────────────────────
    banner("1. 模型管理器单例")
    print(f"  models 实例: {models!r}")
    print(f"  类型: {type(models).__name__}")

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
        print("  下载模型:")
        print("    models.download('embedding')  — 下载 Embedding 模型")
        print("    models.download_all()         — 下载全部模型")

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
    print("  模型存储在: local_models/{org}/{model_name}/")

    # ── 5. 下载配置 ─────────────────────────────────────────────
    banner("5. 模型下载配置")

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

    # ── 6. 下载策略 ─────────────────────────────────────────────
    banner("6. 下载策略")

    print("  可用策略:")
    print(f"    AutoStrategy — 自动选择 HF/ModelScope")
    print(f"    HfStrategy   — HuggingFace Hub（国内可用 hf-mirror.com 镜像）")
    print(f"    MsStrategy   — ModelScope（国内速度更优）")
    print(f"  当前策略: {settings.model.download_source}")
    print(f"  切换: 修改 config/{{env}}.yaml → model.download_source = 'modelscope'")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 模型状态与配置演示完成")
    print()
    print("  下一步: 02_finetune.py — 微调模型管理")


if __name__ == "__main__":
    main()
