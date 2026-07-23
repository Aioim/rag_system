"""python -m scripts 入口 — 等价于 python scripts/setup.py"""
import sys
from pathlib import Path

# 确保 scripts/ 的父目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "scripts"))

from scripts.setup import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 快速部署向导")
    parser.add_argument("--non-interactive", action="store_true",
                        help="非交互模式（从环境变量读取配置）")
    args, _ = parser.parse_known_args()
    main(non_interactive=args.non_interactive)
