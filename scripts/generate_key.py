"""生成 Fernet 密钥文件 (environments/.secret_key)

用法:
    python scripts/generate_key.py              # 交互模式
    python scripts/generate_key.py --force       # 强制覆盖已有密钥
    python scripts/generate_key.py --output /path/to/key  # 自定义路径
"""

import argparse
import sys
from pathlib import Path

# ---- 将项目根目录加入 sys.path，以便导入 src 模块 ----
from _common import PROJECT_ROOT, banner, fail, info, ok, warn

sys.path.insert(0, str(PROJECT_ROOT))


def run(force: bool = False, output_path: str | None = None) -> bool:
    """生成 Fernet 密钥文件。

    Args:
        force: 已存在时是否覆盖
        output_path: 自定义输出路径（默认 PROJECT_ROOT/environments/.secret_key）

    Returns:
        True 表示密钥已就绪（生成成功或已存在）
    """
    from cryptography.fernet import Fernet
    from security.secrets_manager import generate_key_file

    key_path = Path(output_path) if output_path else PROJECT_ROOT / "environments" / ".secret_key"

    if key_path.exists() and not force:
        ok(f"密钥已存在: {key_path}")
        return True

    if force and key_path.exists():
        info(f"覆盖已有密钥: {key_path}")

    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        # 原子写入
        key = Fernet.generate_key()
        tmp = key_path.with_suffix(".tmp")
        tmp.write_bytes(key)
        tmp.replace(key_path)
        ok(f"密钥已生成: {key_path}")
        info("请确保 .gitignore 包含 .secret_key 和 *.key")
        return True
    except OSError as e:
        fail(f"密钥生成失败: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Fernet 密钥文件")
    parser.add_argument("--force", action="store_true", help="覆盖已有密钥")
    parser.add_argument("--output", metavar="PATH", help="自定义输出路径")
    args = parser.parse_args()

    banner("生成 Fernet 密钥")
    success = run(force=args.force, output_path=args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
