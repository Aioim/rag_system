"""配置 + 加密 .env 文件中的敏感字段

用法:
    python scripts/encrypt_env.py                          # 交互模式
    python scripts/encrypt_env.py --input .env --output .env.encrypted
    python scripts/encrypt_env.py --non-interactive         # CI 模式（从环境变量读取）
"""

import argparse
import getpass
import os
import re
import shutil
import sys
from pathlib import Path

from _common import PROJECT_ROOT, banner, fail, info, ok, warn

sys.path.insert(0, str(PROJECT_ROOT))

# .env 中的敏感字段（需要加密的 key）
_SENSITIVE_KEYS = [
    ("LLM_API_KEY", "DeepSeek API 密钥 (https://platform.deepseek.com/)"),
    ("HUGGINGFACE_TOKEN", "HuggingFace Token (https://huggingface.co/settings/tokens)"),
]

_KEY_PATH = PROJECT_ROOT / "environments" / ".secret_key"


def _create_env_from_example(env_path: Path) -> bool:
    """从 .env.example 复制模板"""
    example = PROJECT_ROOT / ".env.example"
    if not example.exists():
        fail(f"模板文件不存在: {example}")
        return False
    shutil.copy(example, env_path)
    info(f"已从 .env.example 复制 → {env_path.name}")
    return True


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    interactive: bool = True,
) -> bool:
    """配置并加密 .env 文件。

    Args:
        input_path: 输入 .env 路径（默认 PROJECT_ROOT/.env）
        output_path: 输出路径（默认覆盖 input_path）
        interactive: True=交互式输入, False=从环境变量读取

    Returns:
        True 表示 .env 已就绪
    """
    env_path = Path(input_path) if input_path else PROJECT_ROOT / ".env"
    out_path = Path(output_path) if output_path else env_path

    # Step 1: 确保 .env 存在
    if not env_path.exists():
        warn(f"{env_path.name} 不存在")
        if not _create_env_from_example(env_path):
            return False

    # Step 2: 检查密钥文件是否存在
    if not _KEY_PATH.exists():
        fail("密钥文件不存在，请先运行: python scripts/generate_key.py")
        return False

    # Step 3: 收集敏感值
    replacements: dict[str, str] = {}
    for key, description in _SENSITIVE_KEYS:
        if interactive:
            try:
                val = getpass.getpass(f"  请输入 {key} ({description}): ").strip()
            except KeyboardInterrupt:
                print("\n  已跳过")
                continue
            if val:
                replacements[key] = val
                info(f"  {key}: ****")
            else:
                info(f"  {key}: 已跳过")
        else:
            val = os.getenv(key, "")
            if val:
                replacements[key] = val
                info(f"  {key}: 从环境变量读取")
            else:
                info(f"  {key}: 未设置，已跳过")

    if not replacements:
        warn("未输入任何敏感值，.env 未加密")
        return True  # 不算失败，可能已手动填写

    # Step 4: 替换占位符为明文
    content = env_path.read_text(encoding="utf-8")
    for key, val in replacements.items():
        placeholder = re.compile(rf"^({re.escape(key)})=.*$", re.MULTILINE)
        content = placeholder.sub(f"{key}={val}", content)
    env_path.write_text(content, encoding="utf-8")

    # Step 5: 加密 .env 中的敏感字段
    try:
        from security.env_encryptor import process_env_file

        process_env_file(str(env_path), str(out_path))

        # Step 6: 如果输出路径不同于输入路径，将加密结果复制回输入路径
        if out_path.resolve() != env_path.resolve():
            shutil.copy2(str(out_path), str(env_path))
            info(f"已将加密内容复制回 {env_path.name}")

        return True
    except Exception as e:
        fail(f"加密失败: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="配置并加密 .env 文件")
    parser.add_argument("--input", metavar="PATH", help="输入 .env 路径")
    parser.add_argument("--output", metavar="PATH", help="输出路径")
    parser.add_argument("--non-interactive", action="store_true", help="非交互模式")
    args = parser.parse_args()

    banner("配置 .env 文件")
    success = run(
        input_path=args.input,
        output_path=args.output,
        interactive=not args.non_interactive,
    )
    if success:
        info("\n.env 配置完成。运行以下命令检查:")
        info("  python scripts/start_server.py")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
