"""本地 LLM OpenAI 兼容代理服务 CLI 入口

用法:
    python -m model.proxy --port 8080
    python -m model.proxy --port 8080 --model /path/to/model.gguf
    python -m model.proxy --host 0.0.0.0 --port 8080
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="本地 LLM OpenAI 兼容代理服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m model.proxy --port 8080
  python -m model.proxy --model local_models/Qwen3-0.6B-Q4_K_M.gguf
  python -m model.proxy --host 0.0.0.0 --port 8080

LangChain 集成:
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
""",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")
    parser.add_argument("--model", default=None, help="GGUF 模型路径（覆盖配置文件）")
    parser.add_argument("--n-ctx", type=int, default=None, help="上下文窗口大小")
    args = parser.parse_args()

    # 添加 src 目录到 sys.path（确保 config 等模块可导入）
    src_path = Path(__file__).resolve().parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # 如果指定了 model 参数，覆盖配置
    if args.model:
        from config import settings
        settings.apply_overrides(f"inference.gguf_file={args.model}")
    if args.n_ctx:
        from config import settings
        settings.apply_overrides(f"inference.n_ctx={args.n_ctx}")

    import uvicorn
    from model.proxy.server import app

    print(f"Local LLM Proxy starting on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"Endpoint:  http://{args.host}:{args.port}/v1/chat/completions")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
