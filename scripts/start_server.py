"""启动 RAG API 服务

用法:
    python scripts/start_server.py                        # 默认 0.0.0.0:8000
    python scripts/start_server.py --port 8080 --reload    # 开发模式
    python scripts/start_server.py --check                 # 仅检查，不启动
"""

import argparse
import subprocess
import sys

from _common import PROJECT_ROOT, banner, fail, info, ok, warn

sys.path.insert(0, str(PROJECT_ROOT))


def check_readiness() -> bool:
    """检查启动就绪状态"""
    all_ok = True

    # 1. 密钥文件
    key_path = PROJECT_ROOT / "environments" / ".secret_key"
    if key_path.exists():
        ok(f"密钥文件: {key_path}")
    else:
        fail(f"密钥文件不存在: {key_path}")
        info("  请先运行: python scripts/generate_key.py")
        all_ok = False

    # 2. .env 文件
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        ok(f"环境配置: {env_path}")
    else:
        warn(f".env 不存在: {env_path}")
        info("  请先运行: python scripts/encrypt_env.py")

    # 3. 模型
    import os
    os.environ.setdefault("ENV", "dev")
    try:
        from model import models
        status = models.status()
        for mt, downloaded in status.items():
            if downloaded:
                ok(f"模型 {mt}: 已就绪")
            else:
                warn(f"模型 {mt}: 未下载")
                info(f"  请运行: python scripts/download_models.py --type {mt}")
                all_ok = False
    except ImportError:
        warn("无法检查模型状态（依赖未安装）")

    return all_ok


def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """启动 API 服务（阻塞调用）"""
    check_readiness()

    info(f"\n启动服务: http://{host}:{port}")
    info("按 Ctrl+C 停止\n")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",  # FastAPI app 入口
        "--host", host,
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    except KeyboardInterrupt:
        info("\n服务已停止")
    except FileNotFoundError:
        fail("uvicorn 未安装，请运行: pip install uvicorn[standard]")


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 RAG API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="端口 (默认: 8000)")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--check", action="store_true", help="仅检查就绪状态，不启动")
    args = parser.parse_args()

    banner("启动 RAG API 服务")

    if args.check:
        check_readiness()
        return

    run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
