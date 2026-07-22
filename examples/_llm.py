"""
examples/_llm.py — 示例共享 LLM 工厂

所有需要 LLM 的演示文件通过此模块获取 ChatOpenAI 实例。
使用 DeepSeek API（OpenAI 兼容协议），需在 .env 中配置 LLM_API_KEY。

使用方式：
    from _llm import create_llm
    llm = create_llm(temperature=0)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402


def create_llm(temperature: float = 0, model: str | None = None):
    """创建 ChatOpenAI 实例（DeepSeek API，OpenAI 兼容协议）

    Args:
        temperature: LLM 温度参数，默认 0（确定性输出）
        model: 模型名称，默认使用 settings.llm.default（deepseek-v4-pro）

    Returns:
        ChatOpenAI 实例，满足 LLMProtocol（ainvoke 返回 .content 属性）

    Raises:
        RuntimeError: 未设置 LLM_API_KEY 环境变量
    """
    from langchain_openai import ChatOpenAI

    api_key = settings.llm.api_key.get_secret_value()
    if not api_key:
        raise RuntimeError(
            f"未设置 {settings.llm.api_key_env} 环境变量！\n"
            f"请在项目根目录的 .env 文件中设置:\n"
            f"  {settings.llm.api_key_env}=sk-xxx\n\n"
            f"支持 Fernet 加密格式:\n"
            f"  {settings.llm.api_key_env}=ENC[base64_ciphertext]\n"
            f"加密工具: python -m security.env_encryptor encrypt <value>"
        )

    return ChatOpenAI(
        model=model or settings.llm.default,
        base_url=settings.llm.api_base_url or "https://api.deepseek.com/v1",
        api_key=api_key,
        temperature=temperature,
    )
