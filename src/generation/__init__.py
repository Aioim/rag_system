"""生成层 — Prompt 组装 / 模型路由 / 生成 / 事实核查 / 引用构建"""
import threading

from generation.citation_builder import CitationBuilder
from generation.fact_checker import FactChecker
from generation.layer import GenerationLayer
from generation.llm_router import LLMRouter
from generation.prompt_assembler import PromptAssembler
from logger import logger
from models.llm import LLMProtocol

# 全局单例
_generation_layer: GenerationLayer | None = None
_lock = threading.Lock()
_init_llm_id: int | None = None


def get_generation_layer(llm: LLMProtocol) -> GenerationLayer:
    """获取生成层全局单例

    首次调用时用传入的 llm 初始化单例。
    后续调用若传入不同 llm 对象，会记录警告但仍返回已缓存的实例。
    """
    global _generation_layer, _init_llm_id

    # 快速路径：已初始化，无锁检查
    if _generation_layer is not None:
        if id(llm) != _init_llm_id:
            logger.warning(
                "get_generation_layer 已初始化，忽略不同的 llm 参数"
            )
        return _generation_layer

    with _lock:
        # 双重检查：可能另一个线程刚完成初始化
        if _generation_layer is None:
            _generation_layer = GenerationLayer(llm)
            _init_llm_id = id(llm)
        elif id(llm) != _init_llm_id:
            logger.warning(
                "get_generation_layer 已初始化，忽略不同的 llm 参数"
            )
        return _generation_layer


def reset_generation_layer() -> None:
    """重置全局单例（测试用）"""
    global _generation_layer, _init_llm_id
    with _lock:
        _generation_layer = None
        _init_llm_id = None


__all__ = [
    "CitationBuilder",
    "FactChecker",
    "GenerationLayer",
    "LLMRouter",
    "PromptAssembler",
    "get_generation_layer",
    "reset_generation_layer",
]
