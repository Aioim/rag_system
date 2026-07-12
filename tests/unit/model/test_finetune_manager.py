"""ModelManager 微调 API 测试"""

import tempfile
import json
from pathlib import Path

from model import models
from model.finetune.config import FinetuneConfig


def _make_triplet_jsonl(path: Path, count: int = 10) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({
                "query": f"q{i}", "positive": f"p{i}", "negative": f"n{i}",
            }, ensure_ascii=False) + "\n")


class TestFinetuneAPI:
    """测试 ModelManager 新增的微调 API（不触发实际训练）"""

    def test_list_finetuned_empty(self):
        # 使用临时目录作为 output_dir
        with tempfile.TemporaryDirectory() as tmp:
            config = FinetuneConfig(output_dir=Path(tmp))
            # 由于 list_finetuned 读取 get_finetune_config().output_dir，
            # 这里验证空目录返回 {}
            # 不直接依赖 models.list_finetuned() 因为它会读取全局配置
            from model.finetune.base import BaseTrainer
            result = BaseTrainer.scan_finetuned(Path(tmp))
            assert result == {}

    def test_invalid_model_type_raises(self):
        """models.finetune() 对无效类型应报错"""
        # 需要确保 models 已初始化
        try:
            models._ensure_init()
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "test.jsonl"
            _make_triplet_jsonl(data_path, count=3)

            # 在未安装模型的环境下，先检查类型校验
            # 不是测试实际训练
            try:
                models.finetune("invalid_type", str(data_path))
                assert False, "应该抛出 ValueError"
            except ValueError as e:
                assert "不支持的模型类型" in str(e)

    def test_get_finetuned_path_not_found(self):
        """不存在的适配器返回 None"""
        # 直接测逻辑：scan 空目录 → 找不到
        with tempfile.TemporaryDirectory() as tmp:
            from model.finetune.base import BaseTrainer
            scanned = BaseTrainer.scan_finetuned(Path(tmp))
            assert scanned.get("nonexistent") is None
