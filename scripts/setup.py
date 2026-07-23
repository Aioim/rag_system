"""RAG 知识库问答系统 — 快速部署向导

用法:
    python scripts/setup.py                    # 交互模式
    python scripts/setup.py --non-interactive   # CI 模式（从环境变量读取）
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from _common import PROJECT_ROOT, banner, fail, info, ok, warn, check_python_version

sys.path.insert(0, str(PROJECT_ROOT))


def _step_check_python() -> bool:
    """Step 1: 检查 Python 版本"""
    is_ok, version_str = check_python_version()
    if is_ok:
        ok(version_str)
        return True
    else:
        fail(f"{version_str} (需要 >= 3.11)")
        return False


def _step_check_deps() -> bool:
    """Step 2: 检查依赖（尝试导入核心模块）"""
    core_modules = [
        ("fastapi", "API 框架"),
        ("langchain", "RAG 编排"),
        ("pydantic", "配置管理"),
        ("cryptography", "加密"),
    ]
    all_ok = True
    for mod, desc in core_modules:
        try:
            __import__(mod)
            ok(desc)
        except ImportError:
            fail(f"{desc} ({mod} 未安装)")
            all_ok = False

    if not all_ok:
        info("\n请运行以下命令安装依赖:")
        info('  pip install -e ".[retrieval,ingestion,fallback]"')
        info("或安装全部:")
        info('  pip install -e ".[all]"')
        return False
    return True


def _step_generate_key(force: bool = False) -> bool:
    """Step 3: 生成密钥"""
    from generate_key import run as gen_key
    return gen_key(force=force)


def _step_setup_env(interactive: bool = True) -> bool:
    """Step 4: 配置 .env"""
    from encrypt_env import run as enc_env
    return enc_env(interactive=interactive)


def _step_download_models() -> bool:
    """Step 5: 下载模型"""
    from download_models import run as dl_models
    return dl_models(model_type="all")


def _step_start_server() -> None:
    """Step 6: 询问是否启动"""
    try:
        answer = input("\n是否启动 API 服务? [y/N]: ").strip().lower()
    except KeyboardInterrupt:
        info("\n已跳过")
        return
    if answer in ("y", "yes"):
        from start_server import run as start
        start()


def main(non_interactive: bool = False) -> None:
    banner("RAG 企业级知识库问答系统 — 快速部署向导")

    steps = [
        ("检查 Python 环境", _step_check_python, True),
        ("检查依赖", _step_check_deps, False),
        ("生成密钥文件", lambda: _step_generate_key(force=False), False),
        ("配置 .env 文件", lambda: _step_setup_env(interactive=not non_interactive), False),
        ("下载模型", _step_download_models, False),
    ]

    total = len(steps)
    failed_steps: list[str] = []

    for i, (name, fn, fatal) in enumerate(steps, 1):
        info(f"\n[{i}/{total}] {name}...")
        try:
            success = fn()
            if not success:
                failed_steps.append(name)
                if fatal:
                    fail(f"{name} 失败，无法继续")
                    sys.exit(1)
        except KeyboardInterrupt:
            warn(f"{name} 已取消")
            info("部署中断。可重新运行继续: python scripts/setup.py")
            sys.exit(0)
        except Exception as e:
            fail(f"{name}: {e}")
            failed_steps.append(name)
            if fatal:
                sys.exit(1)

    # 总结
    banner("部署完成")
    if failed_steps:
        warn(f"以下步骤未完成: {', '.join(failed_steps)}")
        info("请修正后重新运行: python scripts/setup.py")
    else:
        ok("所有步骤完成")

    info("\n启动服务:")
    info("  python scripts/start_server.py")
    info("\n其他命令:")
    info("  python scripts/generate_key.py --force   # 重新生成密钥")
    info("  python scripts/encrypt_env.py            # 重新配置 .env")
    info("  python scripts/download_models.py        # 重新下载模型")
    info("  python scripts/start_server.py --reload  # 开发模式启动")

    if not non_interactive:
        _step_start_server()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 快速部署向导")
    parser.add_argument("--non-interactive", action="store_true", help="非交互模式")
    args = parser.parse_args()
    main(non_interactive=args.non_interactive)
