"""
训练数据加载 & 验证 — JSONL 格式，每种模型类型有独立的 schema
"""

import json
from pathlib import Path
from typing import Optional


class DataValidationError(ValueError):
    """训练数据格式校验失败"""

    def __init__(self, message: str, line_number: Optional[int] = None):
        loc = f" (第 {line_number} 行)" if line_number is not None else ""
        super().__init__(f"数据格式错误{loc}: {message}")


def load_jsonl(path: Path) -> list[dict]:
    """加载 JSONL 文件为 dict 列表，空行自动跳过"""
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    if not path.suffix == ".jsonl":
        raise DataValidationError(f"文件格式必须为 .jsonl，实际为: {path.suffix}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise DataValidationError(f"JSON 解析失败: {e}", line_no)
            records.append(record)

    if not records:
        raise DataValidationError("数据文件为空")
    return records


def validate_embedding_data(records: list[dict]) -> None:
    """校验 Embedding 三元组格式: query, positive, negative"""
    required = {"query", "positive", "negative"}
    for i, r in enumerate(records, start=1):
        missing = required - set(r.keys())
        if missing:
            raise DataValidationError(f"缺少字段: {missing}", i)
        for field in required:
            if not isinstance(r[field], str) or not r[field].strip():
                raise DataValidationError(f"字段 '{field}' 不能为空", i)


def validate_reranker_data(records: list[dict]) -> None:
    """校验 Reranker 二分类格式: query, document, label (0/1)"""
    for i, r in enumerate(records, start=1):
        for field in ("query", "document"):
            if field not in r or not isinstance(r[field], str) or not r[field].strip():
                raise DataValidationError(f"字段 '{field}' 缺失或为空", i)
        if "label" not in r or r["label"] not in (0, 1):
            raise DataValidationError(
                f"字段 'label' 必须为 0 或 1，实际为: {r.get('label')}", i
            )


def validate_llm_data(records: list[dict]) -> None:
    """校验 LLM 指令格式: instruction, input, output（input 可为空字符串）"""
    for i, r in enumerate(records, start=1):
        if "instruction" not in r or not isinstance(r["instruction"], str) or not r["instruction"].strip():
            raise DataValidationError("字段 'instruction' 缺失或为空", i)
        if "input" not in r or not isinstance(r["input"], str):
            raise DataValidationError("字段 'input' 缺失或类型错误", i)
        if "output" not in r or not isinstance(r["output"], str) or not r["output"].strip():
            raise DataValidationError("字段 'output' 缺失或为空", i)


VALIDATORS = {
    "embedding": validate_embedding_data,
    "reranker": validate_reranker_data,
    "llm": validate_llm_data,
}


def split_train_eval(
    records: list[dict], eval_ratio: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """将数据按比例切分为训练集和验证集（保持原始顺序切分）"""
    if not 0.0 <= eval_ratio <= 1.0:
        raise ValueError(f"eval_ratio 必须在 [0, 1] 之间，实际: {eval_ratio}")
    if eval_ratio == 0.0:
        return records, []
    split_idx = max(1, int(len(records) * (1.0 - eval_ratio)))
    return records[:split_idx], records[split_idx:]
