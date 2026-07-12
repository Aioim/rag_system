"""Stage 协议 — 定义 Pipeline 中每个阶段的接口"""

from typing import Protocol

from ingestion.context import PipelineContext


class Stage(Protocol):
    """Pipeline Stage 协议

    每个 Stage 必须提供 name（标识）、fatal（错误是否中断 pipeline）、
    以及 run(ctx) 方法（接收并返回 PipelineContext）。
    """
    name: str
    fatal: bool

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """执行阶段逻辑，接收 ctx 并返回修改后的 ctx"""
        ...
