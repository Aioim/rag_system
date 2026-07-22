"""
02_overrides_env.py — 配置管理：覆盖、重载与环境变量注入

演示内容：
  1. CLI 参数覆盖 (apply_overrides)
  2. 配置热重载 (reload)
  3. 环境变量注入（白名单机制）
  4. Pydantic 模型验证

运行方式：
  cd rag0709
  python examples/01_config/02_overrides_env.py

可选：通过环境变量验证覆盖机制
  # Windows PowerShell
  $env:RETRIEVAL__TOP_K="20"
  python examples/01_config/02_overrides_env.py
  Remove-Item Env:\RETRIEVAL__TOP_K

  # Linux/macOS
  RETRIEVAL__TOP_K=20 python examples/01_config/02_overrides_env.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    from config import settings

    # ── 1. CLI 参数覆盖 ────────────────────────────────────────
    banner("1. CLI 参数覆盖 (apply_overrides)")

    original_top_k = settings.retrieval.top_k
    print(f"  覆盖前: top_k = {original_top_k}")
    settings.apply_overrides("retrieval.top_k=50;retrieval.rrf_k=80;retrieval.mmr_lambda=0.9")
    print(f"  覆盖后: top_k = {settings.retrieval.top_k}")
    print(f"          rrf_k = {settings.retrieval.rrf_k}")
    print(f"          mmr_lambda = {settings.retrieval.mmr_lambda}")
    print(f"  语法: key=value;key2=value2（分号分隔）")

    # ── 2. 配置热重载 ──────────────────────────────────────────
    banner("2. 配置热重载 (reload)")

    print(f"  重载前: top_k = {settings.retrieval.top_k}")
    settings.reload()
    print(f"  重载后: top_k = {settings.retrieval.top_k}")
    print(f"  说明: reload() 会清空 CLI 覆盖，恢复为 YAML 原始值")

    # ── 3. 环境变量注入（白名单） ──────────────────────────────
    banner("3. 环境变量注入（白名单机制）")

    print(f"  ENV 环境变量:      {os.environ.get('ENV', '(未设置)')}")
    print(f"  DEBUG 环境变量:    {os.environ.get('DEBUG', '(未设置)')}")
    print(f"  LLM__DEFAULT:      {os.environ.get('LLM__DEFAULT', '(未设置)')}")
    print()
    print("  环境变量命名规则 (双层嵌套):")
    print("    RETRIEVAL__TOP_K=10        → retrieval.top_k = 10")
    print("    LLM__DEFAULT=deepseek-pro → llm.default = 'deepseek-pro'")
    print("    RAG__CUSTOM__KEY=val      → custom.key = 'val' (RAG__ 逃生前缀)")
    print()
    print("  白名单过滤: 仅识别以配置段根名开头的双层变量")
    print("  系统环境变量 (PATH/TEMP/OS 等) 不会再误入配置")

    # ── 4. Pydantic 模型验证 ───────────────────────────────────
    banner("4. Pydantic v2 配置模型验证")

    print("  所有配置段都经过 Pydantic v2 校验：")
    print(f"  ✓ retrieval.top_k >= 1:           {settings.retrieval.top_k}")
    print(f"  ✓ retrieval.mmr_lambda ∈ [0, 1]:  {settings.retrieval.mmr_lambda}")
    print(f"  ✓ chunking.chunk_size > 0:        {settings.chunking.chunk_size}")
    print(f"  ✓ generation.dedup_threshold ∈ [0,1]: {settings.generation.dedup_threshold}")
    print()
    print("  配置优先级: CLI覆盖 > 环境变量 > {env}.yaml > 代码默认值")

    # ── 总结 ───────────────────────────────────────────────────
    banner("✅ 配置覆盖与重载演示完成")
    print()
    print("  环境切换: ENV=prod python examples/01_config/02_overrides_env.py")
    print("  新增配置: 在 config/{env}.yaml 中添加新段，对应 Pydantic 模型")


if __name__ == "__main__":
    main()
