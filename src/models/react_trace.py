"""ReAct Agent 单步推理记录"""
from dataclasses import dataclass


@dataclass
class ReActTrace:
    """ReAct Agent 单步推理记录"""
    iteration: int
    thought: str
    action: str          # "search" | "web_search" | "finish"
    query: str | None = None
    observation: str | None = None
    elapsed_ms: float = 0.0
